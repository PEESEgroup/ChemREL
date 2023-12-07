# `chemrel`

ChemREL Command Line Interface (CLI).

Automate and transfer chemical data extraction using span categorization and relation extraction models.

To initialize the assets required by the CLI, run the following command.

```console
$ chemrel init
```

**Usage**:

```console
$ chemrel [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `aux`: Run one of a number of auxiliary data...
* `clean`: Removes intermediate files to start data...
* `init`: Initializes files required by package at...
* `predict`: Predicts the spans and/or relations in a...
* `rel`: Configure and/or train a relation...
* `span`: Configure and/or train a span...

## `chemrel aux`

Run one of a number of auxiliary data processing commands.

**Usage**:

```console
$ chemrel aux [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `extract-elsevier-paper`: Converts Elsevier paper with specified DOI...
* `extract-paper`: Converts paper PDF at specified path into...

### `chemrel aux extract-elsevier-paper`

Converts Elsevier paper with specified DOI code into a sequence of JSONL files each corresponding to a text
chunk, where each JSONL line is tokenized by sentence. Example: if provided path is `dir/file` and the Paper text
contains two chunks, files `dir/file_1.jsonl` and `dir/file_2.jsonl` will be generated; otherwise, if the Paper
text contains one chunk, `dir/file.jsonl` will be generated.

**Usage**:

```console
$ chemrel aux extract-elsevier-paper [OPTIONS] DOI_CODE API_KEY JSONL_PATH
```

**Arguments**:

* `DOI_CODE`: DOI code of paper, not in URL form  [required]
* `API_KEY`: Elsevier API key  [required]
* `JSONL_PATH`: Filepath to save JSONL files to, ignores filename extension  [required]

**Options**:

* `--char-limit INTEGER`: Character limit of each text chunk in generated Paper object
* `--help`: Show this message and exit.

### `chemrel aux extract-paper`

Converts paper PDF at specified path into a sequence of JSONL files each corresponding to a text chunk, where each
    JSONL line is tokenized by sentence. Example: if provided path is `dir/file` and the Paper text contains two
    chunks, files `dir/file_1.jsonl` and `dir/file_2.jsonl` will be generated; otherwise, if the Paper text contains
    one chunk, `dir/file.jsonl` will be generated.

**Usage**:

```console
$ chemrel aux extract-paper [OPTIONS] PAPER_PATH JSONL_PATH
```

**Arguments**:

* `PAPER_PATH`: File path of paper PDF  [required]
* `JSONL_PATH`: Filepath to save JSONL files to, ignores filename extension  [required]

**Options**:

* `--char-limit INTEGER`: Character limit of each text chunk in generated Paper object
* `--help`: Show this message and exit.

## `chemrel clean`

Removes intermediate files to start data preparation and training from a clean slate.

**Usage**:

```console
$ chemrel clean [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `chemrel init`

Initializes files required by package at given path.

**Usage**:

```console
$ chemrel init [OPTIONS] [PATH]
```

**Arguments**:

* `[PATH]`: File path in which to initialize required files  [default: ./]

**Options**:

* `--help`: Show this message and exit.

## `chemrel predict`

Predicts the spans and/or relations in a given text using the given models.

**Usage**:

```console
$ chemrel predict [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `rel`: Predicts spans and the relations between...
* `span`: Predicts spans contained in given text and...

### `chemrel predict rel`

Predicts spans and the relations between them contained in given text determined by the given models and prints
them.

**Usage**:

```console
$ chemrel predict rel [OPTIONS] SC_MODEL_PATH REL_MODEL_PATH TEXT
```

**Arguments**:

* `SC_MODEL_PATH`: File path of span categorization model to be used  [required]
* `REL_MODEL_PATH`: File path of relation extraction model to be used  [required]
* `TEXT`: Text content to predict spans within  [required]

**Options**:

* `--help`: Show this message and exit.

### `chemrel predict span`

Predicts spans contained in given text and prints them.

**Usage**:

```console
$ chemrel predict span [OPTIONS] SC_MODEL_PATH TEXT
```

**Arguments**:

* `SC_MODEL_PATH`: File path of span categorization model to be used  [required]
* `TEXT`: Text content to predict spans within  [required]

**Options**:

* `--help`: Show this message and exit.

## `chemrel rel`

Configure and/or train a relation extraction model.

**Usage**:

```console
$ chemrel rel [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `process-data`: Parses the gold-standard annotations from...
* `test`: Applies the best relation extraction model...
* `tl-cpu`: Trains the relation extraction (rel) model...
* `tl-gpu`: Trains the relation extraction (rel) model...
* `train-cpu`: Trains the relation extraction (rel) model...
* `train-gpu`: Trains the relation extraction (rel) model...

### `chemrel rel process-data`

Parses the gold-standard annotations from the Prodigy annotations.

**Usage**:

```console
$ chemrel rel process-data [OPTIONS]
```

**Options**:

* `--annotations-file TEXT`: File path of Prodigy annotations  [default: assets/goldrels.jsonl]
* `--train-file TEXT`: File path of training data corpus  [default: reldata/train.spacy]
* `--dev-file TEXT`: File path of dev corpus  [default: reldata/dev.spacy]
* `--test-file TEXT`: File path of test data corpus  [default: reldata/test.spacy]
* `--help`: Show this message and exit.

### `chemrel rel test`

Applies the best relation extraction model to unseen text and measures accuracy at different thresholds.

**Usage**:

```console
$ chemrel rel test [OPTIONS]
```

**Options**:

* `--trained-model TEXT`: File path of trained model to be used  [default: reltraining/model-best]
* `--test-file TEXT`: File path of test data corpus  [default: reldata/test.spacy]
* `--help`: Show this message and exit.

### `chemrel rel tl-cpu`

Trains the relation extraction (rel) model using transfer learning on the CPU and evaluates it on the dev corpus.

**Usage**:

```console
$ chemrel rel tl-cpu [OPTIONS]
```

**Options**:

* `--tl-tok2vec-config TEXT`: File path of config file for Tok2Vec span categorization model  [default: configs/rel_TL_tok2vec.cfg]
* `--train-file TEXT`: File path of training data corpus  [default: reldata/train.spacy]
* `--dev-file TEXT`: File path of dev corpus  [default: reldata/dev.spacy]
* `--help`: Show this message and exit.

### `chemrel rel tl-gpu`

Trains the relation extraction (rel) model with a Transformer using transfer learning on the GPU and evaluates it
on the dev corpus.

**Usage**:

```console
$ chemrel rel tl-gpu [OPTIONS]
```

**Options**:

* `--tl-trf-config TEXT`: File path of config file for transformer span categorization model  [default: configs/rel_TL_trf.cfg]
* `--train-file TEXT`: File path of training data corpus  [default: reldata/train.spacy]
* `--dev-file TEXT`: File path of dev corpus  [default: reldata/dev.spacy]
* `--gpu-id TEXT`: The GPU device identifier to be used  [default: 0]
* `--help`: Show this message and exit.

### `chemrel rel train-cpu`

Trains the relation extraction (rel) model on the CPU and evaluates it on the dev corpus.

**Usage**:

```console
$ chemrel rel train-cpu [OPTIONS]
```

**Options**:

* `--tok2vec-config TEXT`: File path of config file for Tok2Vec span categorization model  [default: configs/rel_tok2vec.cfg]
* `--train-file TEXT`: File path of training data corpus  [default: reldata/train.spacy]
* `--dev-file TEXT`: File path of dev corpus  [default: reldata/dev.spacy]
* `--help`: Show this message and exit.

### `chemrel rel train-gpu`

Trains the relation extraction (rel) model with a Transformer on the GPU and evaluates it on the dev corpus.

**Usage**:

```console
$ chemrel rel train-gpu [OPTIONS]
```

**Options**:

* `--trf-config TEXT`: File path of config file for transformer span categorization model  [default: configs/rel_trf.cfg]
* `--train-file TEXT`: File path of training data corpus  [default: reldata/train.spacy]
* `--dev-file TEXT`: File path of dev corpus  [default: reldata/dev.spacy]
* `--gpu-id TEXT`: The GPU device identifier to be used  [default: 0]
* `--help`: Show this message and exit.

## `chemrel span`

Configure and/or train a span categorization model.

**Usage**:

```console
$ chemrel span [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `process-data`: Instructs to use the Prodigy function...
* `test`: Applies the best span categorization model...
* `tl-cpu`: Trains the span categorization (sc) model...
* `tl-gpu`: Trains the span categorization (sc) model...
* `train-cpu`: Trains the span categorization (sc) model...
* `train-gpu`: Trains the span categorization (sc) model...

### `chemrel span process-data`

Instructs to use the Prodigy function (data-to-spacy) for data processing.

**Usage**:

```console
$ chemrel span process-data [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `chemrel span test`

Applies the best span categorization model to unseen text and measures accuracy at different thresholds.

**Usage**:

```console
$ chemrel span test [OPTIONS]
```

**Options**:

* `--trained-model TEXT`: File path of trained model to be used  [default: sctraining/model-best]
* `--test-file TEXT`: File path of test data corpus  [default: scdata/test.spacy]
* `--gpu-id TEXT`: The GPU device identifier to be used
* `--help`: Show this message and exit.

### `chemrel span tl-cpu`

Trains the span categorization (sc) model using transfer learning on the CPU and evaluates it on the dev corpus.

**Usage**:

```console
$ chemrel span tl-cpu [OPTIONS]
```

**Options**:

* `--tl-tok2vec-config TEXT`: File path of config file for Tok2Vec span categorization model  [default: configs/sc_TL_tok2vec.cfg]
* `--train-file TEXT`: File path of training data corpus  [default: scdata/train.spacy]
* `--dev-file TEXT`: File path of dev corpus  [default: scdata/dev.spacy]
* `--help`: Show this message and exit.

### `chemrel span tl-gpu`

Trains the span categorization (sc) model using transfer learning on the GPU and evaluates it on the dev corpus.

**Usage**:

```console
$ chemrel span tl-gpu [OPTIONS]
```

**Options**:

* `--tl-trf-config TEXT`: File path of config file for transformer span categorization model  [default: configs/sc_TL_trf.cfg]
* `--train-file TEXT`: File path of training data corpus  [default: scdata/train.spacy]
* `--dev-file TEXT`: File path of dev corpus  [default: scdata/dev.spacy]
* `--gpu-id TEXT`: The GPU device identifier to be used  [default: 0]
* `--help`: Show this message and exit.

### `chemrel span train-cpu`

Trains the span categorization (sc) model on the CPU and evaluates it on the dev corpus.

**Usage**:

```console
$ chemrel span train-cpu [OPTIONS]
```

**Options**:

* `--tok2vec-config TEXT`: File path of config file for Tok2Vec span categorization model  [default: configs/sc_tok2vec.cfg]
* `--train-file TEXT`: File path of training data corpus  [default: scdata/train.spacy]
* `--dev-file TEXT`: File path of dev corpus  [default: scdata/dev.spacy]
* `--help`: Show this message and exit.

### `chemrel span train-gpu`

Trains the span categorization (sc) model on the GPU and evaluates it on the dev corpus.

**Usage**:

```console
$ chemrel span train-gpu [OPTIONS]
```

**Options**:

* `--trf-config TEXT`: File path of config file for transformer span categorization model  [default: configs/sc_trf.cfg]
* `--train-file TEXT`: File path of training data corpus  [default: scdata/train.spacy]
* `--dev-file TEXT`: File path of dev corpus  [default: scdata/dev.spacy]
* `--gpu-id TEXT`: The GPU device identifier to be used  [default: 0]
* `--help`: Show this message and exit.
