### Repo for testing ChemREL packaging

-------------------------------------------------

# Project of Span Categorization and Relation Extraction of Chemicals and Their Properties

This project focuses on automating and transferring chemical data extraction using span categorization and relation extraction models.

## To Do
 - Separate the predict 
 - Explain steps##
 - Overriding cli 
 - Documentation
 - Tutorial
 - Testing
 
## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Variables](#variables)
- [Assets](#assets)
- [Workflows](#workflows)
- [Commands](#commands)

## Overview

The project uses span categorization (sc) and relation extraction (rel) models to process and analyze chemical data. The configuration, commands, and workflows for the project are defined in a YAML file.

## Directory Structure

The project consists of the following directories:

- functions: Contains Python scripts for data processing and evaluation.
- configs: Stores configuration files for training the models.
- assets: Holds annotation assets created with Prodigy.
- scdata: Contains data for the span categorization model.
- sctraining: Stores trained models and output files for the span categorization model.
- reldata: Contains data for the relation extraction model.
- reltraining: Stores trained models and output files for the relation extraction model.
- chemrelmodels: Directory for storing chemical relation models.

## Variables

Variables are defined under the 'vars' section in the YAML file. They include file paths for various configurations, data, and trained models.

## Assets

Annotations created with Prodigy are stored in 'assets/goldrels.jsonl'.

## Workflows

Two workflows are defined in the YAML file: 'cpu' and 'gpu'. Both workflows execute a series of commands to process data, train, and test the span categorization (sc) and relation extraction (rel) models. The 'gpu' workflow is for training with a GPU.

## Commands

There are 12 commands defined in the YAML file, with most of them being used in the workflows:

1. sc_process_data: Instructs to use the Prodigy function (data-to-spacy) for data processing.
2. rel_process_data: Parses the gold-standard annotations from the Prodigy annotations.
3. sc_train_cpu: Trains the span categorization (sc) model on the CPU and evaluates it on the dev corpus.
4. sc_TL_cpu: Trains the span categorization (sc) model using transfer learning on the CPU and evaluates it on the dev corpus.
5. sc_train_gpu: Trains the span categorization (sc) model on the GPU and evaluates it on the dev corpus.
6. sc_TL_gpu: Trains the span categorization (sc) model using transfer learning on the GPU and evaluates it on the dev corpus.
7. sc_test: Applies the best span categorization model to unseen text and measures accuracy at different thresholds.
8. rel_train_cpu: Trains the relation extraction (rel) model on the CPU and evaluates it on the dev corpus.
9. rel_TL_cpu: Trains the relation extraction (rel) model using transfer learning on the CPU and evaluates it on the dev corpus.
10. rel_train_gpu: Trains the relation extraction (rel) model with a Transformer on the GPU and evaluates it on the dev corpus.
11. rel_TL_gpu: Trains the relation extraction (rel) model with a Transformer using transfer learning on the GPU and evaluates it on the dev corpus.
12. rel_test: Applies the best relation extraction model to unseen text and measures accuracy at different thresholds.
13. clean: Removes intermediate files to start data preparation and training from a clean slate.
