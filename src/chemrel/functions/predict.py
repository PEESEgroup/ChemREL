import spacy
from spacy import displacy
from spacy.tokens import DocBin, Doc
from chemrel.functions.pipeline import custom_relation_extractor, relation_scorer
from chemrel.functions.model import build_relation_model, build_classification_layer, build_instances, build_tensors
from prettytable import PrettyTable


def predict_span(sc_model, text):
    nlp = spacy.load(sc_model)
    doc = nlp(text)
    spans = doc.spans["sc"]
    spans_table = PrettyTable()
    spans_table.field_names = ["#", "Span", "Label", "Confidence"]
    counter = 1
    for span, confidence in zip(spans, spans.attrs["scores"]):
        spans_table.add_row([counter, span.text, span.label_, confidence])
        counter += 1
    print(spans_table)


def predict_rel(sc_model, rel_model, text):
    nlp = spacy.load(sc_model)
    doc = nlp(text)
    doc.ents = doc.spans['sc']
    nlp2 = spacy.load(rel_model)
    rel_table = PrettyTable()
    rel_table.field_names = []
    counter = 1
    for name, proc in nlp2.pipeline:
        doc = proc(doc)
    for value, rel_dict in doc._.rel.items():
        for e in doc.ents:
            for b in doc.ents:
                if e.start == value[0] and b.start == value[1]:
                    if rel_dict['CHEMP'] >= 0.5:
                        if len(rel_table.field_names) == 0:
                            rel_table.field_names = ["#"] + sorted([e.label_, b.label_]) + ["Confidence"]
                        rel_table.add_row([counter] + [span.text for span in sorted([e, b], key=lambda x: rel_table.field_names[1:3].index(x.label_))] + [rel_dict['CHEMP']])
                        counter += 1
    print(rel_table)