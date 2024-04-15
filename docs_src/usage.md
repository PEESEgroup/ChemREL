# Basic Usage

## Calling the CLI

Before using any ChemREL CLI commands, first, `cd` into the directory in which you initialized ChemREL as
follows, where `[PATH]` is the ChemREL Initial Direcory path in which you originally ran the `chemrel init` command.
```console
$ cd [PATH]
```
```{caution}
If you fail to `cd` into this directory before beginning to use ChemREL in any new terminal session, the necessary files will not be visible to the program.
```

To print the help text for any command or subcommand, simply enter the desired command followed by the `--help` flag.
For example, the following will print the help text for the `chemrel predict` command.
```console
$ chemrel predict --help
```

### Training New Models

ChemREL span categorization, relation extraction, and associated transfer learning models can be trained through the
ChemREL CLI.

For a demonstration on training a span categorization model for a new chemical property, see the
[Span Categorization Demo notebook](notebooks/spancat_demo.ipynb).

For a full list of available CLI commands, view the [ChemREL CLI Reference](cli).

## Importing Functions

In addition to the CLI, the ChemREL PyPI package exposes a number of functions which can be imported from within your own
code.
You must first import the specific submodule containing your desired function from the `functions` package. For example,
to import the `auxiliary` submodule, run the following line.
```python
from chemrel.functions import auxiliary
```

You can then reference any available functions within the `auxiliary` submodule. For example, to call the
`extract_paper()` function, you can run the following.
```python
paper = auxiliary.extract_paper("/example/paper/path")
```
For a demonstration on importing methods in a Jupyter notebook, see the
[Prediction Functions Demo notebook](notebooks/predict_demo.ipynb).

For a full list of importable functions, view the [ChemREL Functions Reference](functions_index).

```{include} ../README.md
:start-after: <!-- begin citation -->
:end-before: <!-- end citation -->
```