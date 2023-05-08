import pathlib
import typer
import os
import chemrel.cli_constants as constants
import chemrel.functions.predict as predict
import chemrel.functions.auxiliary as aux
from pathlib import Path
from huggingface_hub import snapshot_download

app = typer.Typer()

predict_app = typer.Typer()
app.add_typer(predict_app, name="predict")

workflow_app = typer.Typer()
app.add_typer(workflow_app, name="workflow")

span_app = typer.Typer()
app.add_typer(span_app, name="span")

rel_app = typer.Typer()
app.add_typer(rel_app, name="rel")

aux_app = typer.Typer()
app.add_typer(aux_app, name="aux")


# App commands

@app.command("init")
def init(path: str = "./"):
    """
    Initializes necessary files at given path.
    """
    print("Creating directories...")
    for dir in ["configs", "assets", "scdata", "sctraining", "reldata", "reltraining"]:
        Path(f"{os.path.normpath(path)}/{dir}").mkdir(parents=True, exist_ok=True)
    print("Downloading models...")
    snapshot_download(repo_id="AbdulelahAlshehri/chemrelmodels", local_dir= "./")
    print("Complete.")
    if path != "./": print(f"Note: Run `cd {path}` before running other commands.")


@app.command("test")
def test():
    os.system("python --version")


@app.command("clean")
def clean():
    """
    Removes intermediate files to start data preparation and training from a clean slate.
    """
    os.system("rm -rf reldata/*")
    os.system("rm -rf reltraining/*")
    os.system("rm -rf sctraining/*")


# Predict commands

@predict_app.command("span")
def predict_span(
        sc_model_path: str,
        text: str,
):
    predict.predict_span(os.path.normpath(sc_model_path), text)


@predict_app.command("rel")
def predict_rel(
        sc_model_path: str,
        rel_model_path: str,
        text: str,
):
    predict.predict_rel(os.path.normpath(sc_model_path), os.path.normpath(rel_model_path), text)


# Workflow commands

@workflow_app.command("all-gpu")
def workflow_all_gpu():
    """
    Executes a series of commands to process data, train, and test the span categorization (sc) and relation
    extraction (rel) models using the GPU.
    """
    print("Running `span process-data`...")
    span_process_data()
    print("Running `span train-cpu`...")
    span_train_cpu()
    print("Running `span tl-cpu`...")
    span_tl_cpu()
    print("Running `span test`...")
    span_test()
    print("Running `rel process-data`...")
    rel_process_data()
    print("Running `rel train-cpu`...")
    rel_train_cpu()
    print("Running `rel tl-cpu`...")
    rel_tl_cpu()
    print("Running `rel test`...")
    rel_test()
    print("Running `clean`...")
    clean()
    print("Complete.")


@workflow_app.command("all-cpu")
def workflow_all_cpu():
    """
    Executes a series of commands to process data, train, and test the span categorization (sc) and relation
    extraction (rel) models using the CPU.
    """
    print("Running `span process-data`...")
    span_process_data()
    print("Running `span train-gpu`...")
    span_train_gpu()
    print("Running `span tl-gpu`...")
    span_tl_gpu()
    print("Running `span test`...")
    span_test()
    print("Running `rel process-data`...")
    rel_process_data()
    print("Running `rel train-gpu`...")
    rel_train_gpu()
    print("Running `rel tl-gpu`...")
    rel_tl_gpu()
    print("Running `rel test`...")
    rel_test()
    print("Running `clean`...")
    clean()
    print("Complete.")


# Span cat commands

@span_app.command("process-data")
def span_process_data():
    """
    Instructs to use the Prodigy function (data-to-spacy) for data processing.
    """
    print("Use the Prodigy function (data-to-spacy) to complete this step.")


@span_app.command("train-cpu")
def span_train_cpu(
        tok2vec_config: str = constants.SC_TOK2VEC_CONFIG,
        train_file: str = constants.SC_TRAIN_FILE,
        dev_file: str = constants.SC_DEV_FILE,
):
    """
    Trains the span categorization (sc) model on the CPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train ${tok2vec_config} --output sctraining --paths.train ${train_file} --paths"
              f".dev ${dev_file}")


@span_app.command("tl-cpu")
def span_tl_cpu(
        tl_tok2vec_config: str = constants.SC_TL_TOK2VEC_CONFIG,
        train_file: str = constants.SC_TRAIN_FILE,
        dev_file: str = constants.SC_DEV_FILE,
):
    """
    Trains the span categorization (sc) model using transfer learning on the CPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train ${tl_tok2vec_config} --output sctraining --paths.train ${train_file}" 
              f" --paths.dev ${dev_file}")


@span_app.command("train-gpu")
def span_train_gpu(
        trf_config: str = constants.SC_TRF_CONFIG,
        train_file: str = constants.SC_TRAIN_FILE,
        dev_file: str = constants.SC_DEV_FILE,
):
    """
    Trains the span categorization (sc) model on the GPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train ${trf_config} --output sctraining --paths.train ${train_file} --paths.dev" 
              f" ${dev_file} --gpu-id 0")


@span_app.command("tl-gpu")
def span_tl_gpu(
        tl_trf_config: str = constants.SC_TL_TRF_CONFIG,
        train_file: str = constants.SC_TRAIN_FILE,
        dev_file: str = constants.SC_DEV_FILE,
):
    """
    Trains the span categorization (sc) model using transfer learning on the GPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train ${tl_trf_config} --output sctraining --paths.train ${train_file} --paths"
              f".dev ${dev_file} --gpu-id 0")


@span_app.command("test")
def span_test(
        trained_model: str = constants.SC_TRAINED_MODEL,
        test_file: str = constants.SC_TEST_FILE,
):
    """
    Applies the best span categorization model to unseen text and measures accuracy at different thresholds.
    """
    os.system(f"python -m spacy evaluate ${trained_model} ${test_file}")


# Relation commands

@rel_app.command("process-data")
def rel_process_data(
        annotations_file: str = constants.ANNOTATIONS,
        train_file: str = constants.REL_TRAIN_FILE,
        dev_file: str = constants.REL_DEV_FILE,
        test_file: str = constants.REL_TEST_FILE,
):
    """
    Parses the gold-standard annotations from the Prodigy annotations.
    """
    os.system(f"python ./functions/parser.py ${annotations_file} ${train_file} ${dev_file} ${test_file}")


@rel_app.command("train-cpu")
def rel_train_cpu(
        tok2vec_config: str = constants.REL_TOK2VEC_CONFIG,
        train_file: str = constants.REL_TRAIN_FILE,
        dev_file: str = constants.REL_DEV_FILE,
):
    """
    Trains the relation extraction (rel) model on the CPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train ${tok2vec_config} --output reltraining --paths.train ${train_file}"
              f" --paths.dev ${dev_file} -c ./functions/build.py")


@rel_app.command("tl-cpu")
def rel_tl_cpu(
        tl_tok2vec_config: str = constants.REL_TL_TOK2VEC_CONFIG,
        train_file: str = constants.REL_TRAIN_FILE,
        dev_file: str = constants.REL_DEV_FILE,
):
    """
    Trains the relation extraction (rel) model using transfer learning on the CPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train ${tl_tok2vec_config} --output reltraining --paths.train ${train_file}"
              f" --paths.dev ${dev_file} -c ./functions/build.py")


@rel_app.command("train-gpu")
def rel_train_gpu(
        trf_config: str = constants.REL_TRF_CONFIG,
        train_file: str = constants.REL_TRAIN_FILE,
        dev_file: str = constants.REL_DEV_FILE,
):
    """
    Trains the relation extraction (rel) model with a Transformer on the GPU and evaluates it on the dev corpus.
    """
    os.system(f"python -m spacy train ${trf_config} --output reltraining --paths.train ${train_file}"
              f" --paths.dev ${dev_file} -c ./functions/build.py --gpu-id 0")


@rel_app.command("tl-gpu")
def rel_tl_gpu(
        tl_trf_config: str = constants.REL_TL_TRF_CONFIG,
        train_file: str = constants.REL_TRAIN_FILE,
        dev_file: str = constants.REL_DEV_FILE,
):
    """
    Trains the relation extraction (rel) model with a Transformer using transfer learning on the GPU and evaluates it
    on the dev corpus.
    """
    os.system(f"python -m spacy train ${tl_trf_config} --output reltraining --paths.train ${train_file}"
              f" --paths.dev ${dev_file} -c ./functions/build.py --gpu-id 0")


@rel_app.command("test")
def rel_test(
        trained_model: str = constants.REL_TRAINED_MODEL,
        test_file: str = constants.REL_TEST_FILE,
):
    """
    Applies the best relation extraction model to unseen text and measures accuracy at different thresholds.
    """
    os.system(f"python ./functions/test.py ${trained_model} ${test_file} False")


# Auxiliary commands

@aux_app.command("extract-paper")
def aux_extract_paper(
        paper_path: str,
        jsonl_path: str,
        char_limit: int = None,
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
        doi_code: str,
        api_key: str,
        jsonl_path: str,
        char_limit: int = None,
):
    """
    Converts Elsevier paper with specified DOI code into a sequence of JSONL files each corresponding to a text
    chunk, where each JSONL line is tokenized by sentence. Example: if provided path is `dir/file` and the Paper text
    contains two chunks, files `dir/file_1.jsonl` and `dir/file_2.jsonl` will be generated; otherwise, if the Paper
    text contains one chunk, `dir/file.jsonl` will be generated.
    """
    paper = aux.get_elsevier_paper(doi_code, api_key, char_limit)
    paper.write_to_jsonl(jsonl_path)
