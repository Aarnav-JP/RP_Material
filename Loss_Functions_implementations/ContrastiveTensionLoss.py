from __future__ import annotations
import optuna
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys
from torch import nn
import torch
from torch.utils.data import DataLoader
from optuna.trial import TrialState
from sentence_transformers import SentenceTransformer, InputExample, evaluation,losses
from sentence_transformers import evaluation
import os
from sklearn.utils.class_weight import compute_class_weight
import datetime
import time  # Added import for time tracking
from sklearn import preprocessing
from typing import Iterable, Dict
from torch import Tensor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers.util import fullname
from matplotlib.cm import get_cmap
from torch.nn import functional as F
from typing import Tuple,Any

import copy
import math
import random
from collections.abc import Iterable
from sentence_transformers.SentenceTransformer import SentenceTransformer




EPS = 1e-9

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str,default="MR_toy")
parser.add_argument("--batch", type=int)
parser.add_argument("--warm", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--iter", type=int)
parser.add_argument("--sigma", type=float)
# parser.add_argument("--model", type=str)
parser.add_argument("--margin", type=float, default=0.5)  # Add this with other arguments
args = parser.parse_args()

# Model and Data Configuration
# model_name = 'all-mpnet-base-v2'
model_name = 'all-MiniLM-L6-v2'
data = args.data
# model_name = args.model
# data = '20ng'
# data_dir = 'data/' + data + '/'

data_dir = '../Implementation/toy_datasets/' + data + '/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
print("dataset:", data)

# Initialize Model and Training Parameters
model = SentenceTransformer(model_name, device=device)
batch_size = args.batch
step = args.warm
rate = args.lr
ep = args.epochs
iter = args.iter
sigma = args.sigma

# Set Seed for Reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

now = datetime.datetime.now()
program_name = os.path.splitext(os.path.basename(__file__))[0]
sys.stdout = open('outputs/' + data + ' ContrastiveTesnionLoss '+model_name+ '_' + now.strftime("%Y-%m-%d-%H-%M-%S") + '.txt','w')

# A modified SentenceTransformer that stores and prints training loss
class ModifiedSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name, device):
        super(ModifiedSentenceTransformer, self).__init__(model_name)
        self.num_epochs = 0
        self.current_epoch_loss = 0
        self.training_losses = []

    def finalize_epoch(self):
        self.training_losses.append(self.current_epoch_loss)
        print(f"Epoch {self.num_epochs} - Training Loss: {self.current_epoch_loss:.4f}")
        self.current_epoch_loss = 0
        self.num_epochs += 1

# Define a simple distance metric container
class SiameseDistanceMetric:
    EUCLIDEAN = lambda x, y: nn.functional.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: nn.functional.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - nn.functional.cosine_similarity(x, y)

# Our custom ContrastiveLoss
class ContrastiveTensionLoss(nn.Module):


    def __init__(self, model: SentenceTransformer) -> None:
        super().__init__()
        self.model2 = model  # This will be the final model used during the inference time.
        self.model1 = copy.deepcopy(model)
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")


    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features1, sentence_features2 = tuple(sentence_features)
        reps_1 = self.model1(sentence_features1)["sentence_embedding"]  # (bsz, hdim)
        reps_2 = self.model2(sentence_features2)["sentence_embedding"]


        sim_scores = (
            torch.matmul(reps_1[:, None], reps_2[:, :, None]).squeeze(-1).squeeze(-1)
        )  # (bsz,) dot product, i.e. S1S2^T


        loss = self.criterion(sim_scores, labels.type_as(sim_scores))

        # if hasattr(self.model1, "current_epoch_loss"):
        #     self.model1.current_epoch_loss += loss.item()
        if hasattr(self.model2, "current_epoch_loss"):
            self.model2.current_epoch_loss += loss.item()
        return loss

from torch.utils.data import Dataset, DataLoader
from sentence_transformers import InputExample

class ContrastiveTensionDatasetWrapper(Dataset):
    def __init__(self, contrastive_loader):

        self.data = [example for batch in contrastive_loader for example in batch] # Convert generator to list

    def __len__(self):
        """Returns the total number of batches."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves a batch by index."""
        return self.data[idx]

class ContrastiveTensionDataLoader:
    def __init__(self, sentences, batch_size, pos_neg_ratio=8):
        self.sentences = sentences
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio
        self.collate_fn = None

        if self.batch_size % self.pos_neg_ratio != 0:
            raise ValueError(
                f"ContrastiveTensionDataLoader was loaded with a pos_neg_ratio of {pos_neg_ratio} and a batch size of {batch_size}. The batch size must be divisible by the pos_neg_ratio"
            )

    def __iter__(self):
        random.shuffle(self.sentences)
        sentence_idx = 0
        batch = []

        while sentence_idx + 1 < len(self.sentences):
            s1 = self.sentences[sentence_idx]
            if len(batch) % self.pos_neg_ratio > 0:  # Negative (different) pair
                sentence_idx += 1
                s2 = self.sentences[sentence_idx]
                label = 0
            else:  # Positive (identical pair)
                s2 = self.sentences[sentence_idx]
                label = 1

            sentence_idx += 1
            batch.append(InputExample(texts=[s1, s2], label=label))

            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch) if self.collate_fn is not None else batch
                batch = []

    def __len__(self):
        return math.floor(len(self.sentences) / (2 * self.batch_size))

model = ModifiedSentenceTransformer(model_name, device=device)

# Load Data
train = pd.read_csv(data_dir + 'train.csv')
train = train.sample(frac=1).reset_index(drop=True)

val = pd.read_csv(data_dir + 'val.csv')
val = val.sample(frac=1).reset_index(drop=True)

test = pd.read_csv(data_dir + 'test.csv')
test = test.sample(frac=1).reset_index(drop=True)

# Label Encoding
label_encoder = preprocessing.LabelEncoder()
X_train = train['text'].values
y_train = train['label'].values
y_train = label_encoder.fit_transform(y_train)

label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

# Compute Class Weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

X_val = val['text'].values
y_val = val['label'].values
y_val = label_encoder.transform(y_val)
X_test = test['text'].values
y_test = test['label'].values
y_test = label_encoder.transform(y_test)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
num_labels = len(np.unique(y_train))
print(num_labels)


def make_training_examples(X, y):
    train_examples = []
    for i in range(0, len(X) - 1, 2):  # Ensure pairs
        train_examples.append(InputExample(texts=[X[i], X[i+1]], label=y[i]))
    return train_examples

print("No. of training examples:", len(make_training_examples(X_train, y_train)))

def normalize_adj(adj):
    rowsum = torch.sum(adj, dim=1).to_dense()
    rowsum = rowsum.to(torch.complex64)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    ret = adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)
    return ret

def guassian(emb, sigma):
    sq_dists = torch.cdist(emb, emb, p=2) ** 2
    weight = torch.exp(-sq_dists / (2 * sigma ** 2))
    weight = weight - torch.diag(torch.diag(weight))    
    return weight


def modified_lpa(train_emb, test_emb, Ytrain, n_yval, iter, sigma):
    n_val = n_yval
    emb = torch.cat((train_emb, test_emb), dim=0)
    num_nodes = emb.shape[0]
    labels = torch.cat((Ytrain, torch.zeros(test_emb.shape[0]).to(device)), dim=0)

    Y = torch.zeros((num_nodes, num_labels)).to(device)
    Y = Y.to(torch.complex64)
    for k in range(num_labels):
        Y[labels == k, k] = 1
    labels = Y
    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    train_mask[:Ytrain.shape[0]] = 1
    val_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    val_mask[Ytrain.shape[0]:Ytrain.shape[0]+n_val]=1
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    test_mask[Ytrain.shape[0] + n_val:num_nodes] = 1

    emb = emb / emb.norm(dim=1, keepdim=True)
    adj = guassian(emb, sigma)
    adj = adj.to(torch.complex64)
    adj = adj + adj.t()
    adj = normalize_adj(adj)
    adj = adj.to_dense()

    F = torch.zeros_like(Y, dtype=torch.complex64)
    F[train_mask] = Y[train_mask]
    F[val_mask] = Y[val_mask]   
    F[test_mask] = 0

    for i in range(iter):
        F = adj.mm(F)
        F /= torch.sum(F, axis=1, keepdims=True) + EPS
        F[train_mask] = Y[train_mask]

    F = F.real + EPS
    F /= torch.sum(F, axis=1, keepdims=True)

    # do deal with NAN error
    F_val = F[val_mask].real
    F_test = F[test_mask].real
    
    # Replace any remaining NaNs with 0 and ensure finite values
    F_val = F[val_mask].real
    F_test = F[test_mask].real
    
    F_val = np.nan_to_num(F_val.cpu().numpy(), nan=0.0)  # Add .cpu()
    F_test = np.nan_to_num(F_test.cpu().numpy(), nan=0.0)  # Add .cpu()
    
    return torch.from_numpy(F_val), torch.from_numpy(F_test)


class Eval(evaluation.SentenceEvaluator):
    def __init__(self, model, save_path, val_accuracies, val_f1_scores, test_accuracies, test_f1_scores,
                 y_preds, iter, batch_size, sigma,epoch, name: str = "", softmax_model=None,
                 write_csv: bool = True):
        self.epoch = epoch
        self.model = model
        self.save_path = save_path
        self.best_val_acc = -float('inf')  # Initialize with very low value
        self.val_accuracies = val_accuracies
        self.val_f1_scores = val_f1_scores
        self.test_accuracies = test_accuracies
        self.iter = iter
        self.batch_size = batch_size
        self.best_accuracy = 0
        self.best_epoch = -1
        self.last_epoch_time = time.time()  # Initialize the timestamp
        self.sigma = sigma
        self.y_preds = y_preds
        self.test_f1_scores = test_f1_scores
        pass

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        current_time = time.time()
        epoch_time = current_time - self.last_epoch_time
        self.last_epoch_time = current_time
        print(f"Epoch {self.epoch} took {epoch_time:.2f} seconds.")

        train_emb = model.encode(X_train, batch_size=self.batch_size, show_progress_bar=False,
                                 convert_to_tensor=True, device=device, normalize_embeddings=True)
        val_emb = model.encode(X_val, batch_size=self.batch_size, show_progress_bar=False,
                               convert_to_tensor=True, device=device, normalize_embeddings=True)
        test_emb = model.encode(X_test, batch_size=self.batch_size, show_progress_bar=False,
                               convert_to_tensor=True, device=device, normalize_embeddings=True)

        val_ypred, test_ypred = modified_lpa(train_emb, torch.cat((val_emb, test_emb), dim=0), 
                                             ytrain, len(yval), self.iter, self.sigma)

        if self.epoch<11 and self.epoch%2==0:
            F = train_emb.cpu().numpy()
            F = np.nan_to_num(F, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
            Y = y_train
            X_embedded = TSNE(init='random', max_iter=500).fit_transform(F)

            L = np.unique(Y)

            # Dynamically generate colors for up to 25 classes using the updated colormap API
            cmap = plt.colormaps['tab20']  # Use the tab20 colormap for distinct colors

            plt.figure()
            for k in L:
                color = cmap(k / max(L))  # Normalize class index to [0, 1] for the colormap
                plt.scatter(
                    X_embedded[Y == k, 0],
                    X_embedded[Y == k, 1],
                    s=5,
                    label=f'train-label-{k}',
                    marker='o',
                    c=[color]
                )

            # plt.legend()
            plt.axis([-40, 40, -40, 40])
            plt.title('TSNE Visualization of Train Embeddings-ContrastiveTesnionLoss-loss')
            plt.savefig(f'plots/{data}_ContrastiveTesnionLoss_TSNE_train' + str(self.epoch) + '.png')


        val_ypred = val_ypred.cpu().numpy()
        test_ypred = test_ypred.cpu().numpy()
        val_ypred = val_ypred * torch.tensor(class_weights).cpu().numpy()
        test_ypred = test_ypred * torch.tensor(class_weights).cpu().numpy()
        val_ypred = np.argmax(val_ypred, axis=1)
        test_ypred = np.argmax(test_ypred, axis=1)
        val_acc = accuracy_score(yval.cpu().numpy(), val_ypred)
        test_acc = accuracy_score(ytest.cpu().numpy(), test_ypred)
        val_f1 = f1_score(yval.cpu().numpy(), val_ypred, average='macro')
        test_f1 = f1_score(ytest.cpu().numpy(), test_ypred, average='macro')
        self.val_f1_scores.append(val_f1)
        self.val_accuracies.append(val_acc)
        self.y_preds.append(test_ypred)
        self.test_accuracies.append(test_acc)
        self.test_f1_scores.append(test_f1)
        print(f"Epoch:{self.epoch}, Validation Accuracy:{val_acc:.4f}, validation F1-score:{val_f1:.4f}, Test Accuracy:{test_acc:.4f}, Test F1-score:{test_f1:.4f}")
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            self.best_epoch = self.epoch

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            print(f"New best validation accuracy: {val_acc:.4f} at epoch {self.epoch}, saving model...")
            model.save(os.path.join(self.save_path, f"ContrastiveTesnionLoss_loss_{model_name}_{data}"))

        return val_acc

    def on_start(self):
        self.last_epoch_time = time.time()  # Reset timestamp at the start of training

# Prepare Tensors
ytrain = torch.tensor(y_train, dtype=torch.long).to(device)
yval = torch.tensor(y_val, dtype=torch.long).to(device)
ytest = torch.tensor(y_test, dtype=torch.long).to(device)


contrastive_loader = ContrastiveTensionDataLoader(X_train, batch_size=16, pos_neg_ratio=8)

# Wrap it in the PyTorch-compatible dataset
wrapped_dataset = ContrastiveTensionDatasetWrapper(contrastive_loader)

def collate_fn(batch):
    return batch 

train_dataloader = DataLoader(wrapped_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

val_accuracies, val_f1_scores, test_preds, test_accuracies, test_f1_scores =[], [], [], [], []

print('Batch Size:', batch_size, 'Warmup Step:', step, 'Rate:', rate, 'Epoch:', ep,
      'Iter:', iter, 'Sigma:', sigma)

# Initialize Evaluator
save_path = "models/"  

custom_loss_fn = ContrastiveTensionLoss(model=model)  # Replace original loss initialization

# Train the SentenceTransformer Model
start_time = time.time()
train_losses=[]
for epoch in range(ep):
    evaluator = Eval(model, save_path, val_accuracies, val_f1_scores, test_accuracies,test_f1_scores, 
                 test_preds, iter, batch_size, sigma=sigma,epoch=epoch)
    evaluator.on_start()
    model.fit(train_objectives=[(train_dataloader, custom_loss_fn)],
            epochs=1,
            show_progress_bar=True,
            evaluator=evaluator,
            warmup_steps=step,
            optimizer_params={'lr': rate},
        )
    model.finalize_epoch()

end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds.")

# After training, report the best results
max_acc_epoch = val_accuracies.index(max(val_accuracies))
max_f1_score = val_f1_scores[max_acc_epoch]
best_y_pred = test_preds[max_acc_epoch]
best_test_acc = test_accuracies[max_acc_epoch]
best_test_f1 = test_f1_scores[max_acc_epoch]

print('Best Validation Accuracy:', max(val_accuracies), 'at Epoch:', max_acc_epoch)
print('Validation F1-Score at Best Val Accuracy:', max_f1_score)
print("Test Accuracy Corresponding to Best Val Accuracy Epoch:", best_test_acc)
print("F1-score Corresponding to Best Val Accuracy Epoch:", best_test_f1)

with open(f"metrics/{data}_ContrastiveTesnionLoss_val.npy", 'wb') as g2:
   np.save(g2, val_accuracies)
   np.save(g2, val_f1_scores)

with open(f"metrics/{data}_ContrastiveTesnionLoss_test.npy", 'wb') as g3:
   np.save(g3, test_accuracies)
   np.save(g3, test_f1_scores)


def plot_metrics(metric_name, val_metrics, test_metrics, title,ep,save_plot=None):
    epochs = np.arange(ep)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, val_metrics, label=f"Validation {metric_name}", marker='o')
    plt.plot(epochs, test_metrics, label=f"Test {metric_name}", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.show()
    if save_plot:
        plt.savefig(save_plot, format='png', bbox_inches='tight', dpi=300)
   
plot_metrics("Accuracy",  val_accuracies, test_accuracies, "Accuracy Over Epochs",ep,save_plot="plots/"+data+"_"+model_name+"_ContrastiveTesnionLoss_accuracy.png")
plot_metrics("F1 Score",  val_f1_scores, test_f1_scores, "F1 Score Over Epochs",ep,save_plot="plots/"+data+"_"+model_name+"_ContrastiveTesnionLoss_f1scores.png")

# After training is complete, use the accumulated_losses_per_epoch list from the CustomLoss instance
epochs = list(range(1, len(model.training_losses) + 1))
epoch_losses = model.training_losses

np.save(f'embeddings/{data}_{model_name}_train_ContrastiveTesnionLoss.npy',epoch_losses)
# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(epochs, epoch_losses, linestyle='-', color='b')
plt.title(' ContrastiveTesnionLoss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.xticks(epochs)  # Ensure x-axis has ticks for each epoch
plt.tight_layout()
plt.savefig("plots/"+data+"_"+model_name+"_ContrastiveTesnionLoss_losses.png", format='png', bbox_inches='tight', dpi=300)
