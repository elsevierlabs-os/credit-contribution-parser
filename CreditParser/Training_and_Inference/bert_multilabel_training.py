# Databricks notebook source
import datetime
import json
import os
import pickle
import time
from ast import literal_eval

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import transformers
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             multilabel_confusion_matrix)
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelHammingDistance
from transformers import AutoModel, BertModel, BertTokenizer

# COMMAND ----------

# MAGIC %md
# MAGIC #### Sets read/write paths and load data

# COMMAND ----------

DATA_FPATH = "./gpt4_labeled.csv"
OUTPUT_PATH = './'

df = pd.read_csv(DATA_FPATH, index_col=None, converters={'Labels':literal_eval})

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore distribution of labels

# COMMAND ----------

explode_df = df.explode('Labels')
plt.figure(figsize=(10,5))
plt.suptitle("Distribution of Labels in Data", fontsize=16)
countplot = sns.countplot(data=explode_df, x="Labels")
countplot.set_xticklabels(countplot.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

# COMMAND ----------

CLASSES = ["Conceptualization", "Methodology", "Software", "Validation", "Formal analysis", "Investigation", "Resources",
           "Data Curation", "Writing - Original Draft", "Writing - Review & Editing",
            "Visualization", "Supervision", "Project administration", "Funding acquisition", "Other"]

# COMMAND ----------

len(CLASSES)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Convert target labels into binarized list

# COMMAND ----------

def vectorize_labels(df_y):
    """
    Transform labels into one-hot encoded vector representation
    """
    mlb = MultiLabelBinarizer()
    vectorized_y = mlb.fit_transform(df_y)

    return mlb, vectorized_y

mlb, vectorized_labels = vectorize_labels(df['Labels'])

df['vectorized_labels'] = vectorized_labels.tolist()

with open(f"{OUTPUT_PATH}/multilabel_binarizer.pkl", "wb") as f:
    pickle.dump(mlb, f)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare dataset

# COMMAND ----------

class CreditDataset(Dataset):
    """
    Custom dataset class that inherits from torch.utils.data.Dataset
    
    Makes use of Hugging Face transformer encode_plus method to:
        -Split sentence into tokens
        -Add special [CLS] and [SEP] tokens
        -Map tokens to IDs
        -Pad/truncate sentences to max length
        -Create attention masks 
 
    Parameters
    ----------

    dataframe: pd.DataFrame used for model training, contains columns 'text' and vectorized labels 'targets'
    max_len: maximum length (in number of tokens) for the inputs to model

    Returns
    _______
    
    dict: Dictionary of encoded ids, mask, token_type_ids, and targets
        
    """
  
    def __init__(self, dataframe, max_len):
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.data = dataframe
        self.text = self.data.contribution
        self.targets = self.data.vectorized_labels
        self.max_len = max_len

    def __len__(self): # overrides __len__ so that len(dataset) returns size of dataset
        return len(self.text)

    def __getitem__(self, index): #override __getitem__ to support indexing
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
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# COMMAND ----------

def split_data(df, test_size):
    """
    Utility function to create train and validation datasets using stratified
    shuffling for multi-label data via iterstrat package
    
    Parameters
    ----------
    
    df: pd.DataFrame that contains modeling data
    test_size: float that designates proportion of data used for testing

    Returns
    -------
    train_dataset: pd.DataFrame that contains training samples
    valid_dataset: pd.DataFrame that contains validation samples
    """  

    x = df['contribution'].tolist()
    y= df['vectorized_labels'].tolist()
    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=42)
    for train_index, valid_index in msss.split(x, y):
        train_dataset, valid_dataset = df.loc[train_index].reset_index(drop=True), df.loc[valid_index].reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(valid_dataset.shape))

    return train_dataset, valid_dataset
train_dataset, valid_dataset = split_data(df, 0.1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create BERT model class with tuned classification layers

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

def prepare_loaders(train_dataset, valid_dataset, max_len, batch_size, num_workers):
    """
    Utility function that uses torch.utils.data.DataLoader class to create iterable over a dataset
    
    Parameters
    ----------
    Calls CreditDataset class to tokenize and encode text. See docstring for more info
        train_dataset
        valid_dataset
        tokenizer
        max_len
    
    batch_size: int for how many samples per batch to load
    num_workers: int for how many subprocesses to use for data loading
    
    Returns
    -------
    training_loader: iterable training dataset
    validation_loader: iterable validation dataset
    """
    

    training_set = CreditDataset(train_dataset, max_len)  

    training_loader = DataLoader(
        training_set,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers)

    validation_set = CreditDataset(valid_dataset, max_len)

    validation_loader = DataLoader(
        validation_set,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers)

    return training_loader, validation_loader

def train(model, training_loader, optimizer, epoch, pos_weight, device):
    """
    For each batch of training data:
        1. Unpack batch from training loader
        2. Isolate batch ids, mask, token_type_ids and target values
        3. Perform a forward pass over data
        4. Clear previously calculated gradients
        5. Calculate loss
        6. Clear gradients
        7. Perform backward pass to calculate gradients
        8. Update parameters and take step using computed gradient
    
    Parameters
    ----------
    
    model: model class object to be trained
    training_loader: iterable dataset used to train
    optimizer: optimizer to be used to update parameters
    epoch: maximum number of epochs to be trained on
    pos_weight: class weighting can be used in loss function
    """
    
    model.train() #set model to training mode
    for batch_idx, data in enumerate(training_loader):   #1 
        # print('epoch', batch_idx)
        ids = data['ids'].to(device, dtype = torch.long)    #2
        mask = data['mask'].to(device, dtype = torch.long)  
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long) 
        targets = data['targets'].to(device, dtype = torch.float)   

        outputs = model(ids, mask, token_type_ids) #3

        optimizer.zero_grad()   #4
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, targets)   #5
        optimizer.zero_grad()  #6
        loss.backward()  #7
        optimizer.step()  #8

        if batch_idx%5000==0:
           print(f'Epoch: {epoch}, batch_id: {batch_idx}, Training Loss:  {loss.item()}')
    
        
def test(model, validation_loader, valid_loss_input, pos_weight, device):
    """
    For each batch of testing data:
        1.Tell pytorch not to both constructing the compute graph during forward pass since only needed for backprop 
        2. Unpack batch from validation loader
        3. Isolate batch ids, mask, token_type_ids and target values
        4. Calculate predictions
        5. Calculate loss
        6. Accumulate validation loss
        7/8. Move labels and predictions to CPU and append to lists
        
    Parameters
    ----------
    
    model: trained model to validate
    validation_loader: iterable dataset used to validate
    valid_loss_input: initial input loss value
    pos_weight: class weighting can be used in loss function
    
    Returns
    -------
    
    val_targets: list of true validation targets
    val_outputs: list of predicated validation targets
    """
    
    model.eval() #put model in evaluation mode
    valid_loss = 0
    valid_loss_min = valid_loss_input
    val_targets = []
    val_outputs = []
    
    with torch.no_grad(): #1
      for batch_idx, data in enumerate(validation_loader, 0):   #2
            ids = data['ids'].to(device, dtype = torch.long)    #3
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)   #4

            loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, targets) #5 #use utility function to convert bytes to torch tensor
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))  #6
            val_targets.extend(targets.cpu().detach().numpy().tolist())  #7
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())   #8
            

            
    print(f'Validation Loss: {valid_loss}')
    
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).'.format(valid_loss_min,valid_loss))
        valid_loss_min = valid_loss
    
    return val_targets, val_outputs, valid_loss_min

def predict_and_score(val_targets, val_outputs, pred_threshold, classes_len):
    """
    Calculate multi-label hamming distance metric to score predicted vs true
    
    Parameters
    ----------
    
    val_targets: list of true validation targets
    val_outputs: list of predicated validation targets
    
    Returns
    -------
    
    """
    
    val_preds = (np.array(val_outputs) > pred_threshold).astype(int)

    val_preds_tensor = torch.Tensor(val_preds)
    val_targets_tensor = torch.Tensor(val_targets)
    micro_hamming_loss = MultilabelHammingDistance(num_labels=classes_len, average='micro')(val_preds_tensor, val_targets_tensor)
    
    return micro_hamming_loss
    
def pos_weights(df, classes_len):
    """
    Calculate weights for each class
    
    Parameters
    ----------
    
    df: pd.DataFrame
    classes_len: int number of classes in dataset
    
    Returns
    -------
     
    torch tensor of length number of classes
    """
    pos_counts = np.array(df["vectorized_labels"].values.tolist()).sum(axis=0)
    n_samples = len(df)
    
    weights = n_samples / (classes_len * pos_counts)

    return tuple(weights)

def run(trial, train_dataset, valid_dataset):
    """
    Train BERT model for fine tuning on data
    
    -Uses Optuna to define and train hyperparameters found in CONFIG
    -Loads training and validation loaders
    -Trains model
    -Returns micro hamming loss that Optuna uses to minimize and tune parameters
    
    Parameters
    ----------
    
    train_dataset: pd.DataFrame that contains training samples
    valid_dataset: pd.DataFrame that contains validation samples
    
    Returns
    -------
    
    f1_score_macro: objective measure
    
    
    """
    CONFIG = {
        'seed': 42,
        'max_len': trial.suggest_categorical('max_len', [64, 128]),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'num_workers': 0,
        'classes_len': len(CLASSES),
        'optimizer': trial.suggest_categorical('optimizer', [torch.optim.Adam, torch.optim.AdamW]),
        'lr': trial.suggest_float('lr', 1e-5, 4e-4),
        'weight_decay': trial.suggest_float('weight_decay', 0, 0.1),
        'n_epochs': trial.suggest_int('n_epochs', 1, 7),
        'pred_threshold': trial.suggest_float('pred_threshold', 0.3, 0.6),
        'pos_weight': trial.suggest_categorical('pos_weight', [None, pos_weights(train_dataset, len(CLASSES))]), #trial no class balancing vs weighting based on ratio of neg vs pos counts for each class
        'dropout': trial.suggest_float('dropout', 0.0, 0.6),
        'device': "cpu"}

    #device = torch.device(CONFIG['device'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIG['device'] = device
    model = BERTClass(CONFIG['dropout'], CONFIG['classes_len'])
    model.to(device)

    if CONFIG['pos_weight'] != None:
        pos_weight = torch.tensor(CONFIG['pos_weight'],dtype=torch.float).to(device, dtype = torch.float)
    else:
        pos_weight = CONFIG['pos_weight']

    torch.manual_seed(CONFIG['seed'])

    training_loader, validation_loader = prepare_loaders(train_dataset, valid_dataset, CONFIG['max_len'], CONFIG['batch_size'], CONFIG['num_workers'])

    optimizer = CONFIG['optimizer'](params =  model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

    valid_loss_input = np.Inf

    for epoch in range(1, CONFIG['n_epochs']+1):

        train(model, training_loader, optimizer, epoch, pos_weight, device)
        val_targets, val_outputs, valid_loss_min = test(model, validation_loader, valid_loss_input, pos_weight, device)
        valid_loss_input = valid_loss_min


    micro_hamming_loss = predict_and_score(val_targets, val_outputs, CONFIG['pred_threshold'], CONFIG['classes_len'])
    print(f"Hamming Loss (Micro) = {micro_hamming_loss}")
    
    return micro_hamming_loss

if __name__ == '__main__':
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=5)
    study = optuna.create_study(sampler=sampler, direction='minimize')
    start = time.time()
    study.optimize(lambda trial: run(trial, train_dataset, valid_dataset), n_trials=10)
    end = time.time()
    print('Time elapsed:', end - start)
    joblib.dump(study, f"{OUTPUT_PATH}/optuna_scibert.pkl")

# COMMAND ----------

optuna_scibert = joblib.load(f"{OUTPUT_PATH}/optuna_scibert.pkl")
best_params = optuna_scibert.best_params

# COMMAND ----------

# MAGIC %md
# MAGIC #### Re-train with best hyperparameters

# COMMAND ----------

def run_best(train_dataset, valid_dataset, best_params):
    """
    Train BERT model for fine tuning on data
    
    -Uses Optuna to define and train hyperparameters found in CONFIG
    -Loads training and validation loaders
    -Trains model
    -Returns hamming score that Optuna uses to minimize and tune parameters
    
    Parameters
    ----------
    
    train_dataset: pd.DataFrame that contains training samples
    valid_dataset: pd.DataFrame that contains validation samples
    
    Returns
    -------
    
    f1_score_macro: objective measure
    
    
    """

    CONFIG = {
        'seed': 42,
        'max_len': 'default',
        'batch_size': 'default',
        'num_workers': 0,
        'classes_len': len(CLASSES),
        'optimizer': 'default',
        'num_warmup_steps':'default',
        'lr': 'default',
        'weight_decay': 'default',
        'n_epochs': 'default',
        'pred_threshold': 'default',
        'pos_weight': 'default',
        'dropout': 'default',
        'device': "cpu"}

    CONFIG.update(best_params)
    joblib.dump(CONFIG, f"{OUTPUT_PATH}/config_scibert.pkl")
    
    model = BERTClass(CONFIG['dropout'], CONFIG['classes_len'])
    device = torch.device(CONFIG['device'])
    model.to(device)
    
    torch.manual_seed(CONFIG['seed'])

    if CONFIG['pos_weight'] != None:
        pos_weight = torch.tensor(CONFIG['pos_weight'],dtype=torch.float).to(device, dtype = torch.float)
    else:
        pos_weight = CONFIG['pos_weight']
    
    training_loader, validation_loader = prepare_loaders(train_dataset, valid_dataset, CONFIG['max_len'], CONFIG['batch_size'], CONFIG['num_workers'])
    
    optimizer = CONFIG['optimizer'](params =  model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    valid_loss_input = np.Inf
    
    for epoch in range(1, CONFIG['n_epochs']+1):
        
        train(model, training_loader, optimizer, epoch, pos_weight, device)
        val_targets, val_outputs, valid_loss_min = test(model, validation_loader, valid_loss_input, pos_weight, device)
        valid_loss_input = valid_loss_min
        
    micro_hamming_loss = predict_and_score(val_targets, val_outputs, CONFIG['pred_threshold'], CONFIG['classes_len'])
    print(f"Hamming Loss (Micro) = {micro_hamming_loss}")

    torch.save(model.state_dict(), f"{OUTPUT_PATH}/SciBERT_credit_state_dict.pt")
    val_preds = (np.array(val_outputs) > CONFIG['pred_threshold']).astype(int)
    return print(classification_report(val_targets, val_preds, target_names=CLASSES))

# COMMAND ----------

run_best(train_dataset, valid_dataset, best_params)