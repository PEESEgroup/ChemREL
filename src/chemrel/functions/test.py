import random
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example


# Import custom functions
from model import build_relation_model, build_classification_layer, build_instances, build_tensors

from relutils import display_output, display_scores

# Main function that accepts pipeline_path, testing_data, and show_output as arguments
def main(pipeline_path: Path, testing_data: Path, show_output: bool):
    # Load the trained NLP pipeline
    language_model = spacy.load(pipeline_path)

    # Load the test data into a DocBin object
    data_bin = DocBin(store_user_data=True).from_disk(testing_data)
    # Convert the data_bin object into a list of documents
    text_data = data_bin.get_docs(language_model.vocab)
    # Initialize an empty list to store the example objects
    example_set = []

    # Loop through each gold document in the test data
    for gold_doc in text_data:
        # Create a predicted document using the tokens from the gold document
        predicted_doc = Doc(
            language_model.vocab,
            words=[token.text for token in gold_doc],
            spaces=[token.whitespace_ for token in gold_doc],
        )
        # Set the entities for the predicted document
        predicted_doc.ents = gold_doc.ents
        # Apply the pipeline processes to the predicted document
        for pipe_name, pipe_proc in language_model.pipeline:
            predicted_doc = pipe_proc(predicted_doc)
        # Add the example (predicted_doc, gold_doc) to the example_set
        example_set.append(Example(predicted_doc, gold_doc))

        # Display output if show_output is True
        if show_output:
            display_output(gold_doc, predicted_doc)

    # Define the score thresholds for evaluation
    score_thresholds = [0.000, 0.05, 0.1, 0.2,0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]

    # Display the scores of the trained model
    print("\nTrained model results:")
    display_scores(example_set, score_thresholds)


if __name__ == "__main__":
    typer.run(main)