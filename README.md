# ChemREL

Automate and transfer chemical data extraction using span categorization and relation extraction models.

## Introduction

ChemREL is a PIP package that allows you to train chemical data extraction models with ease using a suite of models,
configurations, and data processing methods. ChemREL consists of a command line interface (CLI) through which you can
run various commands, as well as a collection of different functions that you can import into your own code.

### Documentation

To install and set up ChemREL, see the [Setup](#setup) section of this file. The following are also provided as additional
documentation.

+ [ChemREL CLI Reference](chemrel/README.md) - Complete documentation of all available CLI commands.
+ [Toy Jupyter Notebook](toy_notebook.ipynb) - Jupyter notebook demonstrating usage of importable functions.

## Setup
To install ChemREL, run the following.
```console
$ pip install chemrel
```

To initialize the assets required by the CLI, run the following command.

```console
$ chemrel init
```
This will download the necessary models and default configuration files.