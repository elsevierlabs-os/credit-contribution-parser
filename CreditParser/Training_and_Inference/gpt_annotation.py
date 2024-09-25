# Databricks notebook source
import os
import openai
from openai import AzureOpenAI
import pandas as pd
import json
import time

import yaml
from tqdm import tqdm

with open('./openai_config.yaml') as f:
    config = yaml.safe_load(f)
    api_key = config['openai']['api_key']
    api_version = config['openai']['api_version']
    api_base = config['openai']['api_base']
    api_type = config['openai']['api_type']


# COMMAND ----------

class OpenAiAnnotation:

    def __init__(self):
        self.client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=api_base)


    def make_response(self, system_prompt, user_prompt):
        retry = 0
        while retry<3:
            try:
                response= self.client.chat.completions.create(
                                                            model = "gpt4",
                                                            messages=[
                                                                    {"role":"system", "content":system_prompt},
                                                                    {"role":"user","content":user_prompt}
                                                                    ],
                                                            temperature=0.1,
                                                            stream=False)
                text = response.choices[0].message.content
                return text
                # return response
            except Exception as e:
                retry += 1
                time.sleep(3)
                if retry==3:
                    print(e)
                    raise e

    def annotate(self, contribution):
        
        user_prompt = """Given a phrase, provide a multi-label classification using the credit dict. 
        If the phrase does not correspond to one of the 14 Credit classes, classify it as 'Other'.
        Do NOT create new categories. Only choose from the 14 categories in credit_dict or 'Other'.
        The answer should be provided in only JSON format containing a key named 'answer' and value as a python list. 
        Ensure the response follows the format {{"answer": ['classification']}}
        Phrase: {contribution}""".format(contribution=contribution)

        system_prompt = """
        Your task is to act as a multi-label classifier to classify phrases with one or more CRedIT author contribution categories. The dictionary credit_dict below contains keys that correspond with the classification labels and the values are their definitions.

        credit_dict = {'Conceptualization':'Ideas; formulation or evolution of overarching research goals and aims',
        'Methodology':'Development or design of methodology; creation of models',
        'Software':'Programming, software development; designing computer programs; implementation of the computer code and supporting algorithms; testing of existing code components',
        'Validation':'Verification, whether as a part of the activity or separate, of the overall replication/ reproducibility of results/experiments and other research outputs',
        'Formal analysis':'Application of statistical, mathematical, computational, or other formal techniques to analyze or synthesize study data',
        'Investigation':'Conducting a research and investigation process, specifically performing the experiments, or data/evidence collection',
        'Resources':'Provision of study materials, reagents, materials, patients, laboratory samples, animals, instrumentation, computing resources, or other analysis tools',
        'Data Curation':'Management activities to annotate (produce metadata), scrub data and maintain research data (including software code, where it is necessary for interpreting the data itself) for initial use and later reuse',
        'Writing - Original Draft':'Preparation, creation and/or presentation of the published work, specifically writing the initial draft (including substantive translation)',
        'Writing - Review & Editing':'Preparation, creation and/or presentation of the published work by those from the original research group, specifically critical review, commentary or revision – including pre-or postpublication stages',
        'Visualization':'Preparation, creation and/or presentation of the published work, specifically visualization/ data presentation',
        'Supervision':'Oversight and leadership responsibility for the research activity planning and execution, including mentorship external to the core team',
        'Project administration':'Management and coordination responsibility for the research activity planning and execution',
        'Funding acquisition':'Acquisition of the financial support for the project leading to this publication'}
        
        """

        try:
            text = self.make_response(system_prompt, user_prompt)
            answer = json.loads(text)
            contributions = answer['answer']
        except Exception as e:
            print(e)
            contributions = ['Error']

        return contributions

# COMMAND ----------

df = pd.read_csv('./gpt_annotation.csv')

# COMMAND ----------

result = []
for ind, row in df.iterrows():
    contributions = row['contribution']
    annotation = OpenAiAnnotation().annotate(contributions)
    result.append(annotation)

df['Labels'] = result

# COMMAND ----------

errors = df[df['Labels'].apply(lambda x: 'Error' in x or len(x) == 0)].index

# COMMAND ----------

from ast import literal_eval
gpt3_df = pd.read_csv('./gpt_labeled.csv', index_col=None, converters={'Labels':literal_eval})

# COMMAND ----------

df.loc[list(errors), 'Labels'] = gpt3_df.loc[list(errors), 'Labels']

# COMMAND ----------

df[df['Labels'].apply(lambda x: 'Error' in x or len(x) == 0)]

# COMMAND ----------

explode_df = df.explode('Labels')
explode_df['Labels'].unique()

# COMMAND ----------

specific_labels = ['Acquisition', 'Writing', 'Acknowledgement', 'Collaboration']
filtered_df = df[~df['Labels'].apply(lambda x: any(label in specific_labels for label in x))]

# COMMAND ----------

filtered_df['Labels'] = filtered_df['Labels'].apply(lambda x: ['Formal analysis' if label == 'Formal Analysis' else label for label in x])

# COMMAND ----------

explode_df = filtered_df.explode('Labels')
explode_df['Labels'].unique()

# COMMAND ----------

filtered_df.to_csv('./gpt4_labeled.csv')