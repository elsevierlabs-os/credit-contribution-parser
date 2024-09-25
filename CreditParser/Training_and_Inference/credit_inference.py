# Databricks notebook source
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "128")

# COMMAND ----------

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, BertTokenizer
import pyspark.sql.functions as f
import pyspark.sql.types as t
import pickle

# COMMAND ----------

cuda = True
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device

# COMMAND ----------

cs_unstructured_corpus_exploded = table("database.table")

# COMMAND ----------

class BERTClass(torch.nn.Module):
    """
    Fine tune a pre-trained BERT model, adjusted to learn our labels
    
    Contains a dropout layer that aims to minimize overfitting and a final linear layer that inputs
    768 dimension features from BERT and returns number of target features
    
    Forward method is used to feed input to BERT
    
    Parameters
    ----------
    dropout: float value to minimize overfitting
    classes_len: number of target labels
    """
    def __init__(self, dropout, classes_len):
        super(BERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.l2 = torch.nn.Dropout(dropout)
        self.l3 = torch.nn.Linear(768, classes_len)
  
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict = False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

# COMMAND ----------

MAX_LEN = 64
CLASSES_LEN = 15
BATCH_SIZE = 256
# BATCH_SIZE = 32
PRED_THRESHOLD = 0.33659690978574053
DROPOUT = 0.4230770207989652

STATE_DICT = 'path_to_state_dict'
MLB_PATH = 'path_to_multilabel_binarizer'

# COMMAND ----------

init_model = BERTClass(DROPOUT, CLASSES_LEN)
init_model.load_state_dict(torch.load(STATE_DICT), strict=False)
model_state = init_model.state_dict()

bc_model_state = sc.broadcast(model_state)

def get_model_for_eval():
    model = BERTClass(DROPOUT, CLASSES_LEN)
    model.load_state_dict(bc_model_state.value)
    model.eval()
    return model


# COMMAND ----------

class InferenceDataset(Dataset):

    def __init__(self, text):
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text = text
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding='max_length',
            return_token_type_ids = True,
            truncation = True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }

# COMMAND ----------

@f.pandas_udf(t.ArrayType(t.StringType()))
def predict_batch_udf(inference_data: pd.Series) -> pd.Series:
    inference_set = InferenceDataset(inference_data)
    inference_loader = DataLoader(inference_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    model = get_model_for_eval()
    model.to(device)

    inference_outputs = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(inference_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)

            inference_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy().tolist())
            # inference_outputs.extend(torch.sigmoid(outputs).tolist())

    with open(MLB_PATH, "rb") as f:
        mlb = pickle.load(f)

    
    inference_preds = (np.array(inference_outputs) > PRED_THRESHOLD).astype(int)
    labels = mlb.inverse_transform(inference_preds)

    return pd.Series(labels)

# COMMAND ----------

cs_unstructured_corpus_inference = cs_unstructured_corpus_exploded.withColumn("inference_output", predict_batch_udf(cs_unstructured_corpus_exploded.contribution))
cs_unstructured_corpus_inference.write.mode("overwrite").format("delta").saveAsTable("database.table")