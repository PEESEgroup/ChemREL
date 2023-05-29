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
ChemREL can be installed via the command line from the PyPI index using the `pip install` command. Once installed, the
ChemREL CLI can be invoked from any directory in the command line using the `chemrel` command.

To install ChemREL, open the command line and run the following.
```console
pip install chemrel
```

Next, run the following to ensure that ChemREL has been properly installed. A help prompt should display.
```console
chemrel --help
```

Before running any model-related commands, ChemREL must be initialized by downloading all necessary model and config
files required by the package. To download the files to your desired directory, first enter the directory where you wish
to save the files by running the `cd` command in the command line as follows, where `[PATH]` should be replaced
with the directory path of your desired location. Note that you will need to `cd` into this directory before using
ChemREL in the future.
```console
cd [PATH]
```
You can then run the `pwd` command to print the path you have entered to verify that you are in the correct directory.

Next, run the following command to download the required files. The file will be downloaded in the folder path you
previously entered. This may take a while to complete.
```console
chemrel init
```
Note: you can also install the files to a path relative to the currently focused directory by passing an additional
argument into the command, e.g. `chemrel init [ALTERNATE PATH]`.

Once the initialization is complete, you are ready to begin using ChemREL.

## Usage
Before using any ChemREL CLI commands, always make sure to `cd` into the directory in which you initialized ChemREL as
follows, where `[PATH]` is the directory path in which you originally ran the `chemrel init` command.
```console
cd [PATH]
```
Otherwise, the necessary files will not be visible to the program.

To print the help text for any command or subcommand, simply enter the desired command followed by the `--help` flag.
For example, the following will print the help text for the `chemrel predict` command.
```console
chemrel predict --help
```