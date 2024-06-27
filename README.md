# Credit Statement Parsing
This repository contains a series of notebooks to turn unstructured credit statements within publications into a structured output.
<br>
Some of this work was done in a Databricks environment, so some use of pyspark methods were applied.

## Usage
The notebooks should be used in a certain order:
1. Preprocess_XML to extract credit statement and author information from full text XML
2. CreditParser to transform and structure the credit statements
3. Training_and_Inference/gpt_annotation to generate a train/test set for multilabel classifier training
4. Training_and_Inference/bert_multilabel_training to train a bert model on multilabel classification task
5. Training_and_Inference/credit_inference to perform inference over output from CreditParser
6. Training_and_Inference/credit_inference_cleanup to clean up data

## Citing this Repository

@software{Elsevier_Credit_Contribution_Parser,
author = {Elsevier},
license = {MIT},
title = {{Credit Contribution Parser}},
url = {https://github.com/elsevierlabs-os/credit-contribution-parser}
}