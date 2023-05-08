import spacy
from spacy import displacy
from spacy.tokens import DocBin, Doc
from chemrel.functions.pipeline import custom_relation_extractor, relation_scorer
from chemrel.functions.model import build_relation_model, build_classification_layer, build_instances, build_tensors


def predict_span(sc_model, text):
    nlp = spacy.load(sc_model)
    doc = nlp(text)
    spans = doc.spans["sc"]
    for span, confidence in zip(spans, spans.attrs["scores"]):
        print(span,"Type:{'", span.label_,"': ", confidence, "}")


def predict_rel(sc_model, rel_model, text):
    nlp = spacy.load(sc_model)
    doc = nlp(text)
    doc.ents = doc.spans['sc']
    nlp2 = spacy.load(rel_model)
    for name, proc in nlp2.pipeline:
        doc = proc(doc)
    for value, rel_dict in doc._.rel.items():
        for e in doc.ents:
            for b in doc.ents:
                if e.start == value[0] and b.start == value[1]:
                    if rel_dict['CHEMP'] >= 0.5:
                        print(f" NER Pair: {e.text, b.text} --> relation: {rel_dict}")
