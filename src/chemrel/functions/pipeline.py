# Import necessary libraries and modules

from itertools import islice
from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any
from spacy.scorer import PRFScore
from thinc.types import Floats2d
import numpy
from spacy.training.example import Example
from thinc.api import Model, Optimizer
from spacy.tokens.doc import Doc
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.vocab import Vocab
from spacy import Language
from thinc.model import set_dropout_rate
from wasabi import Printer

# Create a Printer object for displaying messages
msg = Printer()

# Set a custom extension 'rel' for relation extraction on Doc objects
Doc.set_extension("rel", default={}, force=True)

# Define the custom_relation_extractor factory function for the Language class
@Language.factory(
    "relation_extractor",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.rel"],
    default_score_weights={
        "rel_micro_p": None,
        "rel_micro_r": None,
        "rel_micro_f": None,
    },
)
def custom_relation_extractor(
    nlp: Language, name: str, model: Model, *, threshold: float
):
    """Create a RelationExtractor component."""
    return RelationExtractor(nlp.vocab, model, name, threshold=threshold)

# Define the RelationExtractor class, which inherits from TrainablePipe
class RelationExtractor(TrainablePipe):
    """
    RelationExtractor component for spaCy, based on the TrainablePipe class.
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        threshold: float,
    ) -> None:
        """
        Instantiate a RelationExtractor.

        :param vocab: The Vocab object used for this RelationExtractor.
        :param model: The thinc Model object used for this RelationExtractor.
        :param name: The name for this RelationExtractor component.
        :param threshold: The threshold value above which a prediction is considered 'True'.
        """
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"labels": [], "threshold": threshold}

    @property
    def labels(self) -> Tuple[str]:
        """
        Retrieve the labels currently in the component.

        :return: A tuple of labels currently in the component.
        """
        return tuple(self.cfg["labels"])

    @property
    def threshold(self) -> float:
        """
        Get the threshold above which a prediction is considered 'True'.

        :return: The threshold value.
        """
        return self.cfg["threshold"]

    def add_label(self, label: str) -> int:
        """
        Introduce a new label to the pipe.

        :param label: The label to add to the pipe.
        :return: 1 if the label was added successfully, 0 otherwise.
        """
        if not isinstance(label, str):
            raise ValueError("Only strings can be added as labels to the RelationExtractor")
        if label in self.labels:
            return 0
        self.cfg["labels"] = list(self.labels) + [label]
        return 1

    # Method to apply the pipe to a Doc object
    def __call__(self, doc: Doc) -> Doc:
        """
        Applies the relation extraction pipeline to a Doc object.

        Args:
        doc (Doc): the document to process.

        Returns:
        Doc: the processed document.
        """
        # Check if there are any candidate instances in the batch of examples
        total_instances = len(self.model.attrs["get_instances"](doc))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc - returning doc as is.")
            return doc

        # Get predictions and set annotations for the Doc object
        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    # Define the predict method of the RelationExtractor class
    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        """
        Apply the pipeline's model to a batch of Doc objects, without modifying them.

        Args:
        docs (Iterable[Doc]): an iterable of Doc objects to process.

        Returns:
        Floats2d: the predicted scores for the input documents.
        """
        get_instances = self.model.attrs["get_instances"]
        total_instances = sum([len(get_instances(doc)) for doc in docs])
        if total_instances == 0:
            msg.info("Could not determine any instances in any docs - can not make any predictions.")
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    # Define the set_annotations method of the RelationExtractor class
    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """
        Modifies a batch of Doc objects, using pre-computed scores.

        Args:
        docs (Iterable[Doc]): an iterable of Doc objects to modify.
        scores (Floats2d): the pre-computed scores for the input documents.
        """
        c = 0
        get_instances = self.model.attrs["get_instances"]
        for doc in docs:
            for (e1, e2) in get_instances(doc):
                offset = (e1.start, e2.start)
                if offset not in doc._.rel:
                    doc._.rel[offset] = {}
                for j, label in enumerate(self.labels):
                    doc._.rel[offset][label] = scores[c, j]
                c += 1


    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss.

        Args:
        examples (Iterable[Example]): The batch of examples to learn from.
        drop (float, optional): The dropout rate to set for the model. Defaults to 0.0.
        set_annotations (bool, optional): Whether to set relation annotations on the Docs. Defaults to False.
        sgd (Optional[Optimizer], optional): The optimizer to use for updating the model parameters. Defaults to None.
        losses (Optional[Dict[str, float]], optional): The dictionary to store loss information. Defaults to None.

        Returns:
        Dict[str, float]: The dictionary of losses.
        """
        # If losses is not provided, create an empty dictionary
        if losses is None:
            losses = {}

        # If the loss for the current component has not been initialized, set it to 0
        losses.setdefault(self.name, 0.0)

        # Set dropout rate for the model
        set_dropout_rate(self.model, drop)

        # Check if there are any candidate instances in the batch of examples
        total_instances = 0
        for eg in examples:
            total_instances += len(self.model.attrs["get_instances"](eg.predicted))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc.")
            return losses

        # Get predictions and calculate the loss and gradient for the batch of examples
        docs = [eg.predicted for eg in examples]
        predictions, backprop = self.model.begin_update(docs)
        loss, gradient = self.get_loss(examples, predictions)

        # Backpropagate the gradient and update the model parameters using the optimizer
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)

        # Add the loss for the current component to the dictionary of losses
        losses[self.name] += loss

        # If set_annotations is True, set the relation annotations on the Docs
        if set_annotations:
            self.set_annotations(docs, predictions)

        return losses


    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """
        Find the loss and gradient of loss for the batch of documents and their predicted scores.

        Args:
        examples (Iterable[Example]): The batch of examples to compute the loss for.
        scores (Floats2d): The predicted scores for the batch of examples.

        Returns:
        Tuple[float, float]: The mean square error loss and its gradient.
        """
        # Convert the examples to a truth tensor
        truths = self._examples_to_truth(examples)

        # Calculate the gradient of the loss with respect to the predicted scores
        gradient = scores - truths

        # Calculate the mean square error loss
        mean_square_error = (gradient ** 2).sum(axis=1).mean()

        return float(mean_square_error), gradient

    def initialize(
            self,
            get_examples: Callable[[], Iterable[Example]],
            *,
            nlp: Language = None,
            labels: Optional[List[str]] = None,
        ):
            """Initializes the relation extractor for training, using a representative set of data examples."""

            # If labels are provided, add them to the model
            if labels is not None:
                for label in labels:
                    self.add_label(label)
            # Otherwise, extract labels from the example data
            else:
                for example in get_examples():
                    # Get relations from the example's reference object
                    relations = example.reference._.rel
                    # Add each label found in the relation dictionary to the model
                    for indices, label_dict in relations.items():
                        for label in label_dict.keys():
                            self.add_label(label)
            self._require_labels()

            # Get a small subbatch of examples to use as a sample for initializing the model
            subbatch = list(islice(get_examples(), 10))
            # Extract the document samples from the subbatch
            doc_sample = [eg.reference for eg in subbatch]
            # Convert the example relations to a NumPy array for initializing the model
            label_sample = self._examples_to_truth(subbatch)
            # If no relation information is available, raise an error
            if label_sample is None:
                raise ValueError("Call begin_training with relevant entities and relations annotated in "
                                 "at least a few reference examples!")
            # Initialize the model with the document samples and relation samples
            self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(self, examples: List[Example]) -> Optional[numpy.ndarray]:
        """Convert a list of examples to a NumPy array of relation samples for initializing the model."""
        # Check that there are actually any candidate instances in this batch of examples
        nr_instances = 0
        for eg in examples:
            nr_instances += len(self.model.attrs["get_instances"](eg.reference))
        if nr_instances == 0:
            return None

        # Create a NumPy array of zeros for the relation samples
        truths = numpy.zeros((nr_instances, len(self.labels)), dtype="f")
        c = 0
        # Iterate over each example and the instances in the reference document
        for i, eg in enumerate(examples):
            for (e1, e2) in self.model.attrs["get_instances"](eg.reference):
                # Get the gold-standard labels for the instance pair
                gold_label_dict = eg.reference._.rel.get((e1.start, e2.start), {})
                # Set the corresponding index in the relation sample array for each label found
                for j, label in enumerate(self.labels):
                    truths[c, j] = gold_label_dict.get(label, 0)
                c += 1
        # Convert the relation sample array to a NumPy array
        truths = self.model.ops.asarray(truths)
        return truths

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples."""
        return relation_scorer(examples, self.threshold)

def relation_scorer(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
    """Score a batch of examples."""
    prf = PRFScore()
    
    # Iterate through each example
    for x in examples:
        # Get the gold and predicted relation labels
        gold_rels = x.reference._.rel
        pred_rels = x.predicted._.rel
        
        # Iterate through each key in the predicted relations dictionary
        for k, v in pred_rels.items():
            gold_k = [key for (key, val) in gold_rels.get(k, {}).items() if val == 1.0]
            
            # Iterate through each predicted label
            for label, score in v.items():
                # Check if the predicted score is above the threshold
                if score >= threshold:
                    # If the predicted label is in the gold labels, increment true positives
                    if label in gold_k:
                        prf.tp += 1
                    # If the predicted label is not in the gold labels, increment false positives
                    else:
                        prf.fp += 1
                # If the predicted score is below the threshold
                else:
                    # If the predicted label is in the gold labels, increment false negatives
                    if label in gold_k:
                        prf.fn += 1
    
    # Return the scores as a dictionary
    return {
        "rel_micro_p": prf.precision,
        "rel_micro_r": prf.recall,
        "rel_micro_f": prf.fscore,
    }
