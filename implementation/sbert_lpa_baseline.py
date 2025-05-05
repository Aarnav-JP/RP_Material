from sentence_transformers import SentenceTransformer,InputExample,losses,evaluation,util
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel as gaussian_similarity
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from scipy import sparse as sp
import pickle


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# print all output to a file
import sys
import os
import datetime
now = datetime.datetime.now()
# sys.stdout = open('output_ohsumed'+now.strftime("%Y-%m-%d-%H-%M-%S")+'.txt', 'w')

epochs = 100

batch_size = 256
lpa_iter =  19 
gamma = 0.7
warmup_steps = 14
lr = 5e-05
random_pairs = 20
positive_pairs = 5
EPS = 1e-9

os.environ['WANDB_MODE'] = 'disabled'


model_name = 'all-MiniLM-L6-v2'
data_dir = './toy_datasets/MR_toy/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

model = SentenceTransformer(model_name)
model

train = pd.read_csv(data_dir+'train.csv')
val = pd.read_csv(data_dir+'val.csv')
test = pd.read_csv(data_dir+'test.csv')
print('Train size:', train.shape)
print('Val size:', val.shape)
print('Test size:', test.shape)
train.head()

X_train = train['text'].values
y_train = train['label'].values

X_val = val['text'].values
y_val = val['label'].values

X_test = test['text'].values
y_test = test['label'].values

def make_training_pairs(X,y):
    train_examples = []
    for i in range(len(X)):
        for j in range(i):
            train_examples.append(InputExample(texts=[X[i], X[j]], label=float(y[i]==y[j])))
    return train_examples

print("No. of training pairs:", len(make_training_pairs(X_train,y_train)))

def normalize_adj(adj):
    if(torch.sum(torch.isnan(adj))):
        raise Exception("ADJ1")
    rowsum = torch.sum(adj, dim=1).to_dense()
    rowsum = rowsum.to(torch.complex64)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    if(torch.sum(torch.isnan(d_inv_sqrt))):
        raise Exception("d_inv_sqrt")
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    ret = adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)
    if(torch.sum(torch.isnan(ret))):
        raise Exception("ret")
    return ret

def modified_lpa(train_emb, test_emb, Ytrain):
    n_val = len(test_emb)
    emb = torch.cat((train_emb, test_emb), dim=0)
    num_nodes = emb.shape[0] 
    labels = torch.cat((Ytrain, torch.zeros(n_val).to(device)), dim=0)
    num_labels = int(torch.max(labels) + 1)
    Y = torch.zeros((num_nodes, num_labels)).to(device)
    Y = Y.to(torch.complex64)
    for k in range(num_labels):
        Y[labels == k, k] = 1
    labels = Y
    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    train_mask[:Ytrain.shape[0]] = 1
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    test_mask[Ytrain.shape[0]:Ytrain.shape[0] + n_val] = 1

    if(torch.sum(torch.isnan(emb))):
        raise Exception("emb")
    
    emb = emb / emb.norm(dim=1, keepdim=True)
    adj = emb.mm(emb.t()) #complete graph
    
    if(torch.sum(torch.isnan(adj))):
        raise Exception("adj")
    
    # adj = torch.zeros_like(C)
    # for k in range(num_nodes):
    #     indx = torch.argsort(C[k, k + 1:])
    #     indx = indx[-num_edges:]
    #     adj[k, k + 1 + indx] = C[k, k + 1 + indx]
        
    adj = adj.to(torch.complex64)
    adj = adj + adj.t()
    adj = normalize_adj(adj)
    adj = adj.to_dense()
    
    if(torch.sum(torch.isnan(adj))):
        raise Exception("ADJ")
    
    if(torch.sum(torch.isnan(Y))):
        raise Exception("Y")
    
    F = torch.zeros_like(Y,dtype=torch.complex64)
    F[train_mask] = Y[train_mask]
    F[test_mask] = 0
    
    for i in range(lpa_iter):
        if(torch.sum(torch.isnan(F))):
            raise Exception("F",i)
        F = adj.mm(F)
        F/=torch.sum(F,axis=1,keepdims=True) + EPS
        if(torch.sum(torch.isnan(F))):
            raise Exception("F1",i)
        F[train_mask] = Y[train_mask]
    
    F = F.real + EPS
    F /= torch.sum(F, axis=1, keepdims=True)
    
    return F[test_mask]

train_examples = make_training_pairs(X_train,y_train)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

class Eval(evaluation.SentenceEvaluator):
    def __init__(self, name: str = "", softmax_model=None, write_csv: bool = True):
        pass

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        train_emb = model.encode(X_train, batch_size=batch_size, show_progress_bar=False,convert_to_tensor=True,device=device,normalize_embeddings=True)
        val_emb = model.encode(X_val, batch_size=batch_size, show_progress_bar=False,convert_to_tensor=True,device=device,normalize_embeddings=True)
        nval = len(val_emb)
        ypred = modified_lpa(train_emb, val_emb, ytrain).cpu().numpy()
        ypred = np.argmax(ypred,axis=1)
        acc = accuracy_score(yval.cpu().numpy(),ypred)
        print("Epoch:",epoch, "Steps:",steps, ",Acc:",acc)
        return acc

ytrain = torch.tensor(y_train, dtype=torch.long).to(device)
yval = torch.tensor(y_val, dtype=torch.long).to(device)

model.fit(train_objectives=[(train_dataloader,losses.CosineSimilarityLoss(model))], epochs=epochs,show_progress_bar=True,evaluator=Eval(), warmup_steps=warmup_steps, optimizer_params={'lr': lr})

