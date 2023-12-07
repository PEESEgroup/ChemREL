import spacy
from spacy import displacy
from spacy.tokens import DocBin, Doc
from chemrel.functions.pipeline import custom_relation_extractor, relation_scorer
from chemrel.functions.model import build_relation_model, build_classification_layer, build_instances, build_tensors
from collections import defaultdict


def predict_span(sc_model, text):
    """
    Predicts spans contained in given text determined by the given span categorization model.

    :param sc_model: File path of span categorization model to be used.
    :type sc_model: str
    :param text: Text content to predict spans within.
    :type text: str
    :return: Dictionary with each key corresponding to a span label, and each value being a list of tuples containing
        the span text and prediction confidence score of that label.
    :rtype: dict
    """
    nlp = spacy.load(sc_model)
    doc = nlp(text)
    spans = doc.spans["sc"]
    span_results = defaultdict(list)
    for span, confidence in zip(spans, spans.attrs["scores"]):
        span_results[span.label_].append((span.text, confidence))
    return dict(span_results)


def predict_rel(sc_model, rel_model, text):
    """
    Predicts spans and the relations between them contained in given text determined by the given span categorization
    and relation extraction model.

    :param sc_model: File path of span categorization model to be used.
    :type sc_model: str
    :param rel_model: File path of relation extraction model to be used.
    :type rel_model: str
    :param text: Text content to predict spans and relations within.
    :type text: str
    :return: Dictionary with each key being a tuple of span labels for which a relation exists, and each value being a
        list of tuples containing a tuple of the related span texts and prediction confidence score of that relation.
    :rtype: dict
    """
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
