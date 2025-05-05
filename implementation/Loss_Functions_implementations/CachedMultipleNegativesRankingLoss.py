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
from matplotlib.cm import get_cmap
from torch.nn import functional as F
from typing import Tuple,Any
from sentence_transformers import util
from collections.abc import Iterable
from sentence_transformers.SentenceTransformer import SentenceTransformer
# from BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction

# import files for this loss function only - 
from contextlib import nullcontext
from functools import partial
from typing import Any
from collections.abc import Iterable, Iterator
import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.models import StaticEmbedding



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
data_dir = '../Implementation/toy_datasets/MR_toy/'

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
sys.stdout = open('outputs/' + data + ' CachedMultipleNegativesRankingLoss '+model_name+ '_' + now.strftime("%Y-%m-%d-%H-%M-%S") + '.txt','w')

class ModifiedSentenceTransformer(SentenceTransformer):
    def __init__(self,model_name,device):
        super(ModifiedSentenceTransformer,self).__init__(model_name)
        self.num_epochs = 0
        self.current_epoch_loss = 0
        self.training_losses = [] 
    def finalize_epoch(self):
        """Store the accumulated training loss for the epoch."""
        self.training_losses.append(self.current_epoch_loss)
        print(f"Epoch {self.num_epochs} - Training Loss: {self.current_epoch_loss:.4f}")
        self.current_epoch_loss = 0  # Reset for the next epoch
        self.num_epochs += 1    

class RandContext:

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedMultipleNegativesRankingLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                surrogate.backward()


class CachedMultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        
        super().__init__()
        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "CachedMultipleNegativesRankingLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding. "
                "Consider using MultipleNegativesRankingLoss instead."
            )

        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mini_batch_size = mini_batch_size
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None
        self.show_progress_bar = show_progress_bar

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {k: v[begin:end] for k, v in sentence_feature.items()}
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mbsz, hdim)
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        loss = self.calculate_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        return loss

    def calculate_loss(self, reps: list[list[Tensor]], with_backward: bool = False) -> Tensor:
        """Calculate the cross-entropy loss. No need to cache the gradients."""

        if len(reps) < 2:
            raise ValueError(f"Expected at least 2 elements in reps, but got {len(reps)}.")

        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)

    # Fix: Ensure reps[1:] is not empty
        if len(reps) > 1:
            embeddings_b = torch.cat([torch.cat(r) for r in reps[1:]])  # ((1 + nneg) * bsz, hdim)
        else:
            raise ValueError("No positive or negative samples found in reps[1:].")

        batch_size = len(embeddings_a)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=embeddings_a.device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        loss = sum(losses)
        return loss

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        self.random_states = []  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)

        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least two sentence feature sets, but got {len(sentence_features)}.")
    

        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}



# Initialize SentenceTransformer
# model = SentenceTransformer(model_name, device=device)
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
    for i in range(len(X) - 1):  # Ensure at least one positive example
        train_examples.append(InputExample(texts=[X[i], X[i + 1]], label=y[i]))  # Ensure pairs
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
            plt.title('TSNE Visualization of Train Embeddings-BatchAllTriplet-loss')
            # plt.savefig(f'plots/{data}_single_cosine_loss_TSNE_train' + str(self.epoch) + '.png')


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
#            model.save(os.path.join(self.save_path, f"single_cos_loss_{model_name}_{data}"))

        return val_acc

    def on_start(self):
        self.last_epoch_time = time.time()  # Reset timestamp at the start of training

# Prepare Tensors
ytrain = torch.tensor(y_train, dtype=torch.long).to(device)
yval = torch.tensor(y_val, dtype=torch.long).to(device)
ytest = torch.tensor(y_test, dtype=torch.long).to(device)

# Prepare DataLoader
train_dataloader = DataLoader(make_training_examples(X_train, y_train), shuffle=True, batch_size=batch_size)

val_accuracies, val_f1_scores, test_preds, test_accuracies, test_f1_scores =[], [], [], [], []

print('Batch Size:', batch_size, 'Warmup Step:', step, 'Rate:', rate, 'Epoch:', ep,
      'Iter:', iter, 'Sigma:', sigma)

# Initialize Evaluator
save_path = "models/"  

custom_loss_fn = CachedMultipleNegativesRankingLoss(model=model)  # Replace original loss initialization

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

#with open(f"metrics/{data}_single_cos_loss_val.npy", 'wb') as g2:
 #   np.save(g2, val_accuracies)
 #   np.save(g2, val_f1_scores)

#with open(f"metrics/{data}_single_cos_loss_test.npy", 'wb') as g3:
  #  np.save(g3, test_accuracies)
  #  np.save(g3, test_f1_scores)


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
    # if save_plot:
        # plt.savefig(save_plot, format='png', bbox_inches='tight', dpi=300)
   
#plot_metrics("Accuracy",  val_accuracies, test_accuracies, "Accuracy Over Epochs",ep,save_plot="plots/"+data+"_"+model_name+"_single_cos_loss_accuracy.png")
#plot_metrics("F1 Score",  val_f1_scores, test_f1_scores, "F1 Score Over Epochs",ep,save_plot="plots/"+data+"_"+model_name+"_single_cos_loss_f1scores.png")

# After training is complete, use the accumulated_losses_per_epoch list from the CustomLoss instance
epochs = list(range(1, len(model.training_losses) + 1))
epoch_losses = model.training_losses

#np.save(f'embeddings/{data}_{model_name}_train_single_cosloss.npy',epoch_losses)
# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(epochs, epoch_losses, linestyle='-', color='b')
plt.title(' BatchAllTriplet Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.xticks(epochs)  # Ensure x-axis has ticks for each epoch
plt.tight_layout()
#plt.savefig("plots/"+data+"_"+model_name+"_single_cos_losses.png", format='png', bbox_inches='tight', dpi=300)
