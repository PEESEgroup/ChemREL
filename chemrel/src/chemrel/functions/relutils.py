import random
import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example

# Import custom functions
from chemrel.functions.pipeline import relation_scorer


# Function to display the output for a given gold_doc and predicted_doc
def display_output(gold_doc, predicted_doc):
    print("\nText:", gold_doc.text)
    print("Spans:", [(ent.start, ent.text, ent.label_) for ent in predicted_doc.ents])
    # Loop through the predicted relations
    for key, rel_values in predicted_doc._.rel.items():
        # Get the gold labels for the relations
        gold_labels = [label for (label, value) in gold_doc._.rel[key].items() if value == 1.0]
        if gold_labels:
            print(f"Pair: {key} --> Gold labels: {gold_labels} --> Predicted values: {rel_values}")
    print()

# Function to display the scores for different thresholds
def display_scores(sample_set, score_thresholds):
    for threshold in score_thresholds:
        # Compute the scores for the given threshold
        score_data = relation_scorer(sample_set, threshold)
        # Format the scores for display
        formatted_scores = {key: "{:.2f}".format(value * 100) for key, value in score_data.items()}
        print(f"Threshold {'{:.2f}'.format(threshold)} \t {formatted_scores}")
        
