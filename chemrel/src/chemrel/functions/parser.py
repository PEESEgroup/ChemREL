import json

import typer
from pathlib import Path
import numpy as np
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer
import random

# Creating a new Printer object
msg = Printer()

# Initializing an empty list for the symmetric labels
SYMM_LABELS = []

# Initializing a dictionary for mapping labels
MAP_LABELS = {
    "CHEMP": "CHEMP"                          # Mapping a label to itself
}

# Defining the parser function with four parameters
def parse(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """
    Creating the corpus from the Prodigy annotations.
    """

    # Creating a custom extension attribute called "rel" for the Doc class and initializing it to an empty dictionary
    Doc.set_extension("rel", default={})

    # Creating a new Vocab object
    vocab = Vocab()

    # Creating an empty dictionary for storing training, development, and testing datasets
    docs = {"train": [], "dev": [], "test": []}

    # Creating empty sets for storing article IDs for each dataset
    ids = {"train": set(), "dev": set(), "test": set()}

    # Initializing counts for all instances and positive instances for each dataset to zero
    count_all = {"train": 0, "dev": 0, "test": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0}

    # Opening the JSON file containing the Prodigy annotations
    with json_loc.open("r", encoding="utf8") as jsonfile:

        # Creating a list of line numbers in the file
        num=list(range(sum(1 for line in jsonfile)))

        # Creating a list of shuffled line numbers
        doclist = random.sample(num,len(num))

        # Splitting the shuffled list into training, development, and testing sets
        trainl, devl, testl = np.split(doclist, [int(len(doclist)*0.8), int(len(doclist)*0.9)])

    # Initializing a counter
    i = 0

    # Opening the JSON file again
    with json_loc.open("r", encoding="utf8") as jfile:

        # Creating a list of line numbers in the file
        n = list(range(sum(1 for line in jfile)))

        # Creating a list of shuffled line numbers
        doclist = random.sample(n, len(n))

        # Splitting the shuffled list into training, development, and testing sets
        train_list, dev_list, test_list = np.split(doclist, [int(len(doclist)*0.8), int(len(doclist)*0.9)])

    # Opening the JSON file again
    with json_loc.open("r", encoding="utf8") as jfile:

        # Iterating over each line in the file
        for line in jfile:

            # Loading the line as a JSON object
            ex = json.loads(line)
            
            span_starts = set()
            
            if ex["answer"] == "accept":
             # Initialize pos and neg counters   
                neg = 0
                
                pos = 0
                
                try:
                    # Extract the words and spaces from the current example's tokens
                    words = [t["text"] for t in ex["tokens"]]
                    
                    spaces = [t["ws"] for t in ex["tokens"]]
                       
                    # Create a new Doc object with the words and spaces
                    doc = Doc(vocab, words=words, spaces=spaces)
                    # Extract the spans from the current example
                    spans = ex["spans"]                    
                    entities = []                    
                    end_to_start = {}
                    
                    # Create entities based on the extracted spans and add them to the Doc object
                    for span in spans:
                        entity = doc.char_span(span["start"], span["end"], label=span["label"])                        
                        end_to_start[span["token_end"]] = span["token_start"]                        
                        entities.append(entity)                        
                        span_starts.add(span["token_start"])                        
                    doc.ents = entities
                    
                    # Initialize the rels dictionary for the current example
                    rels = {}
                    
                    for x1 in span_starts:
                        
                        for x2 in span_starts:
                            
                            rels[(x1, x2)] = {}
                            
                    # Extract the relations from the current example and add them to the rels dictionary
                    relations = ex["relations"]
                    
                    for relation in relations:
                        
                        start = end_to_start[relation["head"]]
                        
                        end = end_to_start[relation["child"]]
                        
                        label = relation["label"]
                        
                        label = MAP_LABELS[label]
                        
                        if label not in rels[(start, end)]:
                            
                            rels[(start, end)][label] = 1.0
                            
                            pos += 1
                            
                        if label in SYMM_LABELS:
                            
                            if label not in rels[(end, start)]:
                                
                                rels[(end, start)][label] = 1.0
                                
                                pos += 1
                     # Add negative relations to the rels dictionary for the current example            
                    for x1 in span_starts:
                        
                        for x2 in span_starts:
                            
                            for label in MAP_LABELS.values():
                                
                                if label not in rels[(x1, x2)]:
                                    
                                    neg += 1
                                    
                                    rels[(x1, x2)][label] = 0.0
                    # Add the rels dictionary to the custom extension of the Doc object                
                    doc._.rel = rels
                    # Check if there are positive relations in the current example
                    if pos > 0:
                        article_id = ex["meta"]["source"]
                     # Add the Doc object to the appropriate list depending on its index
                        if i in dev_list:
                            ids["dev"].add(article_id)
                            docs["dev"].append(doc)
                            count_pos["dev"] += pos
                            count_all["dev"] += pos + neg
                            
                        elif i in train_list:
                            ids["train"].add(article_id)
                            docs["train"].append(doc)
                            count_pos["train"] += pos
                            count_all["train"] += pos + neg
                            
                        elif i in test_list:
                            ids["test"].add(article_id)
                            docs["test"].append(doc)
                            count_pos["test"] += pos
                            count_all["test"] += pos + neg
                            
 # If there is a key error, skip the current example and display an error message            
                except KeyError as e:
                    msg.fail(f"Skipping doc because of key error: {e} in {ex['meta']['source']}")
            i=i+1
#Copy to disk
    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"{len(docs['train'])} training sentences from {len(ids['train'])} articles, "
        f"{count_pos['train']}/{count_all['train']} pos instances."
    )

    docbin = DocBin(docs=docs["dev"], store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        f"{len(docs['dev'])} dev sentences from {len(ids['dev'])} articles, "
        f"{count_pos['dev']}/{count_all['dev']} pos instances."
    )

    docbin = DocBin(docs=docs["test"], store_user_data=True)
    docbin.to_disk(test_file)
    msg.info(
        f"{len(docs['test'])} test sentences from {len(ids['test'])} articles, "
        f"{count_pos['test']}/{count_all['test']} pos instances."
    )
