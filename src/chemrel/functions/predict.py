import spacy
from spacy import displacy
from spacy.tokens import DocBin, Doc
from chemrel.functions.pipeline import custom_relation_extractor, relation_scorer
from chemrel.functions.model import build_relation_model, build_classification_layer, build_instances, build_tensors
from collections import defaultdict


def predict_span(sc_model, text):
    nlp = spacy.load(sc_model)
    doc = nlp(text)
    spans = doc.spans["sc"]
    span_results = defaultdict(list)
    for span, confidence in zip(spans, spans.attrs["scores"]):
        span_results[span.label_].append((span.text, confidence))
    return dict(span_results)


def predict_rel(sc_model, rel_model, text):
    nlp = spacy.load(sc_model)
    doc = nlp(text)
    doc.ents = doc.spans['sc']
    nlp2 = spacy.load(rel_model)
    rel_results = dict()
    for name, proc in nlp2.pipeline:
        doc = proc(doc)
    for value, rel_dict in doc._.rel.items():
        for e in doc.ents:
            for b in doc.ents:
                if e.start == value[0] and b.start == value[1]:
                    if rel_dict['CHEMP'] >= 0.5:
                        record = ((e.text, b.text), rel_dict['CHEMP'])
                        if (e.label_, b.label_) not in rel_results:
                            rel_results[(e.label_, b.label_)] = [record]
                        else:
                            rel_results[(e.label_, b.label_)].append(record)
    return rel_results
