import typer
from rich import print
from rich.table import Table
import os
import chemrel.cli_constants as constants
import chemrel.functions.predict as predict
import chemrel.functions.auxiliary as aux
import chemrel.functions.parser as parser
import chemrel.functions.test as test
from pathlib import Path
from shutil import rmtree
from huggingface_hub import snapshot_download


landing_text = """
ChemREL Command Line Interface (CLI).

Automate and transfer chemical data extraction using span categorization and relation extraction models.

To initialize the assets required by the CLI, run the following command: 

$ chemrel init
"""

app = typer.Typer(help=landing_text)

predict_app = typer.Typer(help="Predicts the spans and/or relations in a given text using the given models.")
app.add_typer(predict_app, name="predict")

span_app = typer.Typer(help="Configure and/or train a span categorization model.")
app.add_typer(span_app, name="span")

rel_app = typer.Typer(help="Configure and/or train a relation extraction model.")
app.add_typer(rel_app, name="rel")

aux_app = typer.Typer(help="Run one of a number of auxiliary data processing commands.")
app.add_typer(aux_app, name="aux")


# App commands

@app.command("init")
def init(path: str = typer.Argument(default="./", help="File path in which to initialize required files")):
    """
    Initializes files required by package at given path.
    """
    print("Creating directories...")
    for dir in ["sctraining", "reltraining"]:
        Path(f"{os.path.normpath(path)}/{dir}").mkdir(parents=True, exist_ok=True)
    print("Downloading files...")
    snapshot_download(repo_id="AbdulelahAlshehri/chemrelmodels", local_dir=path,
                      ignore_patterns=["*.md", ".gitattributes"])
    print("Complete.")
    if path != "./": print(f"Note: Run `cd {path}` before running other commands.")


@app.command("clean")
def clean():
    """
    Removes intermediate files to start data preparation and training from a clean slate.
    """
    dirs = ["reldata", "scdata", "reltraining", "sctraining"]
    for d in dirs:
        print(f"Cleaning `{d}`...")
        for path in Path(d).glob("**/*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                rmtree(path)
    print("Cleaned.")


# Predict commands

@predict_app.command("span")
def predict_span(
        sc_model_path: str = typer.Argument(default=..., help=constants.SC_MODEL_PATH_HELP_STRING),
        text: str = typer.Argument(default=..., help=constants.TEXT_HELP_STRING),
):
    """
    Predicts spans contained in given text and prints them.
    """
    span_results = predict.predict_span(os.path.normpath(sc_model_path), text)
    spans_table = Table("#", "Span", "Label", "Confidence")
    counter = 1
    for label in span_results:
        for span in span_results[label]:
            spans_table.add_row(str(counter), span[0], label, str(span[-1]))
            counter += 1
    if counter == 1:
        print("No spans found.")
    else:
        print(spans_table)


@predict_app.command("rel")
def predict_rel(
        sc_model_path: str = typer.Argument(default=..., help=constants.SC_MODEL_PATH_HELP_STRING),
        rel_model_path: str = typer.Argument(default=..., help=constants.REL_MODEL_PATH_HELP_STRING),
        text: str = typer.Argument(default=..., help=constants.TEXT_HELP_STRING),
):
    """
    Predicts spans and the relations between them contained in given text determined by the given models and prints
    them.
    """
    rel_results = predict.predict_rel(os.path.normpath(sc_model_path), os.path.normpath(rel_model_path), text)
    for rel_type in rel_results:
        rel_table = Table("#", rel_type[0], rel_type[1], "Confidence")
        counter = 1
        for rel_pair in rel_results[rel_type]:
            rel_table.add_row(str(counter), rel_pair[0][0], rel_pair[0][1], str(rel_pair[1]))
            counter += 1
        print(rel_table)


# Span cat commands

@span_app.command("process-data")
def span_process_data():
    """
    Instructs to use the Prodigy function (data-to-spacy) for data processing.
    """
    print("Use the Prodigy function (data-to-spacy) to complete this step.")


@span_app.command("train-cpu")
def span_train_cpu(
        tok2vec_config: str = typer.Option(default=constants.SC_TOK2VEC_CONFIG, help=constants.TOK2VEC_CONFIG_HELP_STRING),
        train_file: str = typer.Option(default=constants.SC_TRAIN_FILE, help=constants.TRAIN_FILE_HELP_STRING),
        dev_file: str = typer.Option(default=constants.SC_DEV_FILE, help=constants.DEV_FILE_HELP_STRING),
):
    """
    Trains the span categorization (sc) model on the CPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train {tok2vec_config} --output sctraining --paths.train {train_file} --paths"
              f".dev {dev_file}")


@span_app.command("tl-cpu")
def span_tl_cpu(
        tl_tok2vec_config: str = typer.Option(default=constants.SC_TL_TOK2VEC_CONFIG, help=constants.TOK2VEC_CONFIG_HELP_STRING),
        train_file: str = typer.Option(default=constants.SC_TRAIN_FILE, help=constants.TRAIN_FILE_HELP_STRING),
        dev_file: str = typer.Option(default=constants.SC_DEV_FILE, help=constants.DEV_FILE_HELP_STRING),
):
    """
    Trains the span categorization (sc) model using transfer learning on the CPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train {tl_tok2vec_config} --output sctraining --paths.train {train_file}" 
              f" --paths.dev {dev_file}")


@span_app.command("train-gpu")
def span_train_gpu(
        trf_config: str = typer.Option(default=constants.SC_TRF_CONFIG, help=constants.TRF_CONFIG_HELP_STRING),
        train_file: str = typer.Option(default=constants.SC_TRAIN_FILE, help=constants.TRAIN_FILE_HELP_STRING),
        dev_file: str = typer.Option(default=constants.SC_DEV_FILE, help=constants.DEV_FILE_HELP_STRING),
        gpu_id: str = typer.Option(default=constants.GPU_ID, help=constants.GPU_ID_HELP_STRING),
):
    """
    Trains the span categorization (sc) model on the GPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train {trf_config} --output sctraining --paths.train {train_file} --paths.dev" 
              f" {dev_file} --gpu-id {gpu_id}")


@span_app.command("tl-gpu")
def span_tl_gpu(
        tl_trf_config: str = typer.Option(default=constants.SC_TL_TRF_CONFIG, help=constants.TRF_CONFIG_HELP_STRING),
        train_file: str = typer.Option(default=constants.SC_TRAIN_FILE, help=constants.TRAIN_FILE_HELP_STRING),
        dev_file: str = typer.Option(default=constants.SC_DEV_FILE, help=constants.DEV_FILE_HELP_STRING),
        gpu_id: str = typer.Option(default=constants.GPU_ID, help=constants.GPU_ID_HELP_STRING),
):
    """
    Trains the span categorization (sc) model using transfer learning on the GPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train {tl_trf_config} --output sctraining --paths.train {train_file} --paths"
              f".dev {dev_file} --gpu-id {gpu_id}")


@span_app.command("test")
def span_test(
        trained_model: str = typer.Option(default=constants.SC_TRAINED_MODEL, help=constants.TRAINED_MODEL_HELP_STRING),
        test_file: str = typer.Option(default=constants.SC_TEST_FILE, help=constants.TEST_FILE_HELP_STRING),
        gpu_id: str = typer.Option(default=None, help=constants.GPU_ID_HELP_STRING),
):
    """
    Applies the best span categorization model to unseen text and measures accuracy at different thresholds.
    """
    if gpu_id is None:
        os.system(f"python -m spacy evaluate {trained_model} {test_file}")
    else:
        os.system(f"python -m spacy evaluate {trained_model} {test_file} --gpu-id {gpu_id}")


# Relation commands

@rel_app.command("process-data")
def rel_process_data(
        annotations_file: str = typer.Option(default=constants.ANNOTATIONS, help=constants.ANNOTATIONS_FILE_HELP_STRING),
        train_file: str = typer.Option(default=constants.REL_TRAIN_FILE, help=constants.TRAIN_FILE_HELP_STRING),
        dev_file: str = typer.Option(default=constants.REL_DEV_FILE, help=constants.DEV_FILE_HELP_STRING),
        test_file: str = typer.Option(default=constants.REL_TEST_FILE, help=constants.TEST_FILE_HELP_STRING),
):
    """
    Parses the gold-standard annotations from the Prodigy annotations.
    """
    parser.parse(Path(annotations_file), Path(train_file), Path(dev_file), Path(test_file))


@rel_app.command("train-cpu")
def rel_train_cpu(
        tok2vec_config: str = typer.Option(default=constants.REL_TOK2VEC_CONFIG, help=constants.TOK2VEC_CONFIG_HELP_STRING),
        train_file: str = typer.Option(default=constants.REL_TRAIN_FILE, help=constants.TRAIN_FILE_HELP_STRING),
        dev_file: str = typer.Option(default=constants.REL_DEV_FILE, help=constants.DEV_FILE_HELP_STRING),
):
    """
    Trains the relation extraction (rel) model on the CPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train {tok2vec_config} --output reltraining --paths.train {train_file}"
              f" --paths.dev {dev_file} -c ./functions/build.py")


@rel_app.command("tl-cpu")
def rel_tl_cpu(
        tl_tok2vec_config: str = typer.Option(default=constants.REL_TL_TOK2VEC_CONFIG, help=constants.TOK2VEC_CONFIG_HELP_STRING),
        train_file: str = typer.Option(default=constants.REL_TRAIN_FILE, help=constants.TRAIN_FILE_HELP_STRING),
        dev_file: str = typer.Option(default=constants.REL_DEV_FILE, help=constants.DEV_FILE_HELP_STRING),
):
    """
    Trains the relation extraction (rel) model using transfer learning on the CPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train {tl_tok2vec_config} --output reltraining --paths.train {train_file}"
              f" --paths.dev {dev_file} -c ./functions/build.py")


@rel_app.command("train-gpu")
def rel_train_gpu(
        trf_config: str = typer.Option(default=constants.REL_TRF_CONFIG, help=constants.TRF_CONFIG_HELP_STRING),
        train_file: str = typer.Option(default=constants.REL_TRAIN_FILE, help=constants.TRAIN_FILE_HELP_STRING),
        dev_file: str = typer.Option(default=constants.REL_DEV_FILE, help=constants.DEV_FILE_HELP_STRING),
        gpu_id: str = typer.Option(default=constants.GPU_ID, help=constants.GPU_ID_HELP_STRING),
):
    """
    Trains the relation extraction (rel) model with a Transformer on the GPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train {trf_config} --output reltraining --paths.train {train_file}"
              f" --paths.dev {dev_file} -c ./functions/build.py --gpu-id {gpu_id}")


@rel_app.command("tl-gpu")
def rel_tl_gpu(
        tl_trf_config: str = typer.Option(default=constants.REL_TL_TRF_CONFIG, help=constants.TRF_CONFIG_HELP_STRING),
        train_file: str = typer.Option(default=constants.REL_TRAIN_FILE, help=constants.TRAIN_FILE_HELP_STRING),
        dev_file: str = typer.Option(default=constants.REL_DEV_FILE, help=constants.DEV_FILE_HELP_STRING),
        gpu_id: str = typer.Option(default=constants.GPU_ID, help=constants.GPU_ID_HELP_STRING),
):
    """
    Trains the relation extraction (rel) model with a Transformer using transfer learning on the GPU and evaluates it
    on the dev corpus.
    """
    os.system(f"python -m spacy train {tl_trf_config} --output reltraining --paths.train {train_file}"
              f" --paths.dev {dev_file} -c ./functions/build.py --gpu-id {gpu_id}")


@rel_app.command("test")
def rel_test(
        trained_model: str = typer.Option(default=constants.REL_TRAINED_MODEL, help=constants.TRAINED_MODEL_HELP_STRING),
        test_file: str = typer.Option(default=constants.REL_TEST_FILE, help=constants.TEST_FILE_HELP_STRING),
):
    """
    Applies the best relation extraction model to unseen text and measures accuracy at different thresholds.
    """
    test.test(Path(trained_model), Path(test_file), False)


# Auxiliary commands

@aux_app.command("extract-paper")
def aux_extract_paper(
        paper_path: str = typer.Argument(default=..., help=constants.PAPER_PATH_HELP_STRING),
        jsonl_path: str = typer.Argument(default=..., help=constants.JSONL_PATH_HELP_STRING),
        char_limit: int = typer.Option(default=None, help=constants.CHAR_LIMIT_HELP_STRING),
):
    """
    Converts paper PDF at specified path into a sequence of JSONL files each corresponding to a text chunk, where each
        JSONL line is tokenized by sentence. Example: if provided path is `dir/file` and the Paper text contains two
        chunks, files `dir/file_1.jsonl` and `dir/file_2.jsonl` will be generated; otherwise, if the Paper text contains
        one chunk, `dir/file.jsonl` will be generated.
    """
    paper = aux.extract_paper(paper_path, char_limit)
    paper.write_to_jsonl(jsonl_path)


@aux_app.command("extract-elsevier-paper")
def aux_extract_elsevier_paper(
        doi_code: str = typer.Argument(default=..., help=constants.DOI_CODE_HELP_STRING),
        api_key: str = typer.Argument(default=..., help=constants.API_KEY_HELP_STRING),
        jsonl_path: str = typer.Argument(default=..., help=constants.JSONL_PATH_HELP_STRING),
        char_limit: int = typer.Option(default=None, help=constants.CHAR_LIMIT_HELP_STRING),
):
    """
    Converts Elsevier paper with specified DOI code into a sequence of JSONL files each corresponding to a text
    chunk, where each JSONL line is tokenized by sentence. Example: if provided path is `dir/file` and the Paper text
    contains two chunks, files `dir/file_1.jsonl` and `dir/file_2.jsonl` will be generated; otherwise, if the Paper
    text contains one chunk, `dir/file.jsonl` will be generated.
    """
    paper = aux.get_elsevier_paper(doi_code, api_key, char_limit)
    paper.write_to_jsonl(jsonl_path)
