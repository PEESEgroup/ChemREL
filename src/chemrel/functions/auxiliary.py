import fitz
import os
import json
import jsonlines
import re
import urllib
import httpx
from nltk.tokenize import sent_tokenize


def extract_paper(paper_path, char_limit=None):
    """
    Converts paper PDF at specified path into a Paper object.
    :param paper_path: File path of paper PDF
    :param char_limit: Character limit of each text chunk in generated Paper object, default is None
    :return: Paper object containing text from specified paper PDF, chunked by character limit
    """
    doc = fitz.open(paper_path)
    filename = os.path.basename(paper_path)
    doi = find_doi(doc)
    paper = Paper(filename, doi)
    for page in doc:
        page_text = page.get_text("text").replace('\n', ' ')
        paper.build_text(page_text, char_limit=char_limit)
    return paper


def get_elsevier_paper(doi_code, api_key, char_limit=None):
    """
    Converts Elsevier paper with specified DOI code into a Paper object.
    :param doi_code: DOI code of paper, not in URL form
    :param api_key: Elsevier API key
    :param char_limit: Character limit of each text chunk in generated Paper object, default is None
    :return: Paper object containing text from specified Elsevier paper with given DOI, chunked by character limit
    """
    endpoint = f"https://api.elsevier.com/content/article/doi/{doi_code}"
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": api_key
    }
    r = httpx.get(endpoint, headers=headers)
    if r.status_code != 200:
        raise Exception("API Request Failed")
    else:
        doc = json.loads(r.text)
        content = doc["full-text-retrieval-response"]["originalText"]
        paper = Paper(None, f"https://doi.org/{urllib.parse.quote(doi_code)}")
        paper.build_text(content, char_limit=char_limit)
        return paper


def find_doi(raw_paper):
    """
    Attempts to find DOI link in paper. Relies on assumption that DOI is present within the first page of the paper.
    :param raw_paper: Raw fitz.Paper object
    :return: URL link of paper DOI found on first page of paper, or "DOI NOT FOUND" if no DOI found
    """
    links = list(raw_paper[0].links(kinds=(fitz.LINK_URI,)))
    for link in links[::-1]:
        if "doi.org/" in link["uri"]:
            return link["uri"]
    doi_regex = re.search("^10.\d{4,9}/[-._;()/:A-Z0-9]+$", raw_paper[0].get_text("text"))
    if doi_regex:
        return f"https://doi.org/{doi_regex.group(0)}"
    return "DOI NOT FOUND"


class Paper:
    """
    A class to represent research papers.
    """

    def __init__(self, filename, doi):
        self.filename = filename
        self.doi = doi
        self.text = []

    def __str__(self):
        return f"{self.filename} - [{self.doi}]"

    def build_text(self, subtext, char_limit):
        """
        Appends provided subtext to Paper text content.
        :param subtext: Subtext to append to text content
        :param char_limit: Character limit of each text chunk to be appended
        :return: None
        """
        if char_limit is None:
            if len(self.text) == 0:
                self.text.append(subtext)
            else:
                self.text[-1] += subtext
        else:
            if len(subtext) <= char_limit:
                self.text.append(subtext)
            else:
                for i in range(0, len(subtext), char_limit):
                    self.text.append(subtext[i: i + char_limit])

    def write_to_jsonl(self, jsonl_path_no_ext):
        """
        Outputs text content to a sequence of JSONL files each corresponding to a text chunk, where each
        JSONL line is tokenized by sentence. Example: if provided path is `dir/file` and the Paper text contains two
        chunks, files `dir/file_1.jsonl` and `dir/file_2.jsonl` will be generated; otherwise, if the Paper text contains
        one chunk, `dir/file.jsonl` will be generated.
        :param jsonl_path_no_ext: Filepath to save JSONL files to, excluding filename extension
        :return: None
        """
        if len(self.text) <= 1:
            suffices = [""]
        else:
            suffices = ["_" + str(i) for i in list(range(1, len(self.text) + 1))]
        for suffix in suffices:
            with jsonlines.open(f"{jsonl_path_no_ext}{suffix}.jsonl", mode='w') as writer:
                for block in self.text:
                    sentences = sent_tokenize(block)
                    for sentence in sentences:
                        writer.write({"text": sentence, "meta": {"source": self.doi}})
