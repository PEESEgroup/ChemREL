<!-- begin intro -->

# ChemREL

Automate and transfer chemical data extraction using span categorization and relation extraction models.

## Introduction

ChemREL is a PyPI (pip) package that allows you to train chemical data extraction models with ease using a suite of models,
configurations, and data processing methods. ChemREL consists of a command line interface (CLI) through which you can
run various commands, as well as a collection of different functions that you can import into your own code.

When utilizing or sharing the dataset and models included with ChemREL, kindly note that they are governed under the CC BY NC 4.0 license as imposed
by software and data mining guidelines, permitting only non-commercial applications. They should be used exclusively for research.
<!-- end intro -->
Any alterations to the models, datasets, or functions included with ChemREL must be properly attributed using the [citation](#citation) provided in this documentation.

## Documentation

To view the full installation and usage reference for ChemREL, visit [ChemREL's documentation](https://peesegroup.github.io/ChemREL/).

The reference material on the documentation website can be additionally found as follows.

+ [ChemREL CLI Reference](chemrel/README.md) - Complete documentation of all available CLI commands.
+ [ChemREL Functions Reference](https://peesegroup.github.io/ChemREL/functions_index.html) - Complete documentation of ChemREL Python functions which
are intended to be importable in your own code.
+ [Prediction Functions Demo Notebook](docs_src/notebooks/predict_demo.ipynb) - Jupyter notebook demonstrating usage of importable prediction functions.
+ [Span Categorization Demo Notebook](docs_src/notebooks/spancat_demo.ipynb) - Jupyter notebook demonstrating the training of a new span categorization model.

<!-- begin citation -->

## Citation
Any alterations to the models, datasets, or functions included with ChemREL must be properly attributed according to the following citation.
```
@article{doi:10.1021/acs.jcim.4c00816,
  author = {Alshehri, Abdulelah S. and Horstmann, Kai A. and You, Fengqi},
  title = {Versatile Deep Learning Pipeline for Transferable Chemical Data Extraction},
  journal = {Journal of Chemical Information and Modeling},
  volume = {64},
  number = {15},
  pages = {5888-5899},
  year = {2024},
  doi = {10.1021/acs.jcim.4c00816},
  note = {PMID: 39009039},
  url = {https://doi.org/10.1021/acs.jcim.4c00816},
  eprint = {https://doi.org/10.1021/acs.jcim.4c00816},
  abstract = {Chemical information disseminated in scientific documents offers an untapped potential for deep learning-assisted insights and breakthroughs. Automated extraction efforts have shifted from resource-intensive manual extraction toward applying machine learning methods to streamline chemical data extraction. While current extraction models and pipelines have ushered in notable efficiency improvements, they often exhibit modest performance, compromising the accuracy of predictive models trained on extracted data. Further, current chemical pipelines lack both transferability─where a model trained on one task can be adapted to another relevant task with limited examples─and extensibility, which enables seamless adaptability for new extraction tasks. Addressing these gaps, we present ChemREL, a versatile chemical data extraction pipeline emphasizing performance, transferability, and extensibility. ChemREL utilizes a custom, diverse data set of chemical documents, labeled through an active learning strategy to extract two properties: normal melting point and lethal dose 50 (LD50). The normal melting point is selected for its prevalence in diverse contexts and wider literature, serving as the foundation for pipeline training. In contrast, LD50 evaluates the pipeline’s transferability to an unrelated property, underscoring variance in its biological nature, toxicological context, and units, among other differences. With pretraining and fine-tuning, our pipeline outperforms existing methods and GPT-4, achieving F1-scores of 96.1\% for entity identification and 97.0\% for relation mapping, culminating in an overall F1-score of 95.4\%. More importantly, ChemREL displays high transferability, effectively transitioning from melting point extraction to LD50 extraction with 10 randomly selected training documents. Released as an open-source package, ChemREL aims to broaden access to chemical data extraction, enabling the construction of expansive relational data sets that propel discovery.}
}
```

<!-- end citation -->
