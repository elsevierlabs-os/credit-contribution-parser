# CRediT Statement Parsing
This repository contains a series of notebooks to turn unstructured CRediT statements
and author contribution statements
within the body text of scholarly documents 
into a structured output that can be analyzed.

This is useful since while CRediT statements are increasingly collected in a structured way,
many documents only contain this information in a free-text field.

Moreover,  in the free-text version of statements, sometimes this statement describes a large range of activities undertaken within the project, which in some cases can be mapped to standard CRediT roles, and in others not.

Some of this work was done in a Databricks environment, so some use of PySPARK methods were applied.

It was initially applied to XML records from Science Direct. You can read more about the XML structure that this presumes [here](https://www.elsevier.com/researcher/author/policies-and-guidelines/elsevier-xml-dtds-and-transport-schemas#Schema-5.15)

You can also apply this software to unstructured author contribution statements from other sources by following steps 2-6 of this process.


## Usage
The notebooks should be used in a certain order:

1. `Preprocess_XML` to extract credit statement and author information from full text XML.
2. `CreditParser` to transform and structure the credit statements at a person-level.
3. `Training_and_Inference/gpt_annotation` to generate a train/test set for multilabel classifier training.
4. `Training_and_Inference/bert_multilabel_training` to train a bert model on multilabel classification task.
5. `Training_and_Inference/credit_inference` to perform inference over output from CreditParser.
6. `Training_and_Inference/credit_inference_cleanup` to clean up data.

## Citing this Repository
Please use the following citation in your publications when referencing this repository.

```bibtext
@software{Elsevier_Credit_Contribution_Parser,
    author = {Elsevier},
    license = {MIT},
    title = {{Credit Contribution Parser}},
    url = {https://github.com/elsevierlabs-os/credit-contribution-parser}
}
```

## Questions and contact
- Josh Fisher: j.fisher@elsevier.com
- Kristy James: k.james@elsevier.com