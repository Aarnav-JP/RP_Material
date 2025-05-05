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

# imports for this loss functions - 
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from sentence_transformers.models import StaticEmbedding

logger = logging.getLogger(__name__)

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
sys.stdout = open('outputs/' + data + ' DenoisingAutoEncoderLoss '+model_name+ '_' + now.strftime("%Y-%m-%d-%H-%M-%S") + '.txt','w')

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

class DenoisingAutoEncoderLoss(nn.Module):
    def __init__(
        self, model: SentenceTransformer, decoder_name_or_path: str | None = None, tie_encoder_decoder: bool = True
    ) -> None:

        super().__init__()

        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "DenoisingAutoEncoderLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding."
            )

        self.encoder = model  # This will be the final model used during the inference time.
        self.tokenizer_encoder = model.tokenizer

        encoder_name_or_path = model[0].auto_model.config._name_or_path
        if decoder_name_or_path is None:
            assert (
                tie_encoder_decoder
            ), "Must indicate the decoder_name_or_path argument when tie_encoder_decoder=False!"
        if tie_encoder_decoder:
            if decoder_name_or_path:
                logger.warning("When tie_encoder_decoder=True, the decoder_name_or_path will be invalid.")
            decoder_name_or_path = encoder_name_or_path

        self.tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_name_or_path)
        self.need_retokenization = not isinstance(self.tokenizer_encoder, type(self.tokenizer_decoder))

        decoder_config = AutoConfig.from_pretrained(decoder_name_or_path)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        kwargs_decoder = {"config": decoder_config}
        try:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name_or_path, **kwargs_decoder)
        except ValueError as e:
            logger.error(
                f'Model name or path "{decoder_name_or_path}" does not support being as a decoder. Please make sure the decoder model has an "XXXLMHead" class.'
            )
            raise e
        assert model[0].auto_model.config.hidden_size == decoder_config.hidden_size, "Hidden sizes do not match!"
        if self.tokenizer_decoder.pad_token is None:
            # Needed by GPT-2, etc.
            self.tokenizer_decoder.pad_token = self.tokenizer_decoder.eos_token
            self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

        if len(AutoTokenizer.from_pretrained(encoder_name_or_path)) != len(self.tokenizer_encoder):
            logger.warning(
                "WARNING: The vocabulary of the encoder has been changed. One might need to change the decoder vocabulary, too."
            )

        if tie_encoder_decoder:
            assert not self.need_retokenization, "The tokenizers should be the same when tie_encoder_decoder=True."
            if len(self.tokenizer_encoder) != len(self.tokenizer_decoder):  # The vocabulary has been changed.
                self.tokenizer_decoder = self.tokenizer_encoder
                self.decoder.resize_token_embeddings(len(self.tokenizer_decoder))
                logger.warning(
                    "Since the encoder vocabulary has been changed and --tie_encoder_decoder=True, now the new vocabulary has also been used for the decoder."
                )
            decoder_base_model_prefix = self.decoder.base_model_prefix
            try:
                # Compatibility with transformers <4.40.0
                PreTrainedModel._tie_encoder_decoder_weights(
                    model[0].auto_model,
                    self.decoder._modules[decoder_base_model_prefix],
                    self.decoder.base_model_prefix,
                )
            except TypeError:
                # Compatibility with transformers >=4.40.0
                PreTrainedModel._tie_encoder_decoder_weights(
                    model[0].auto_model,
                    self.decoder._modules[decoder_base_model_prefix],
                    self.decoder.base_model_prefix,
                    encoder_name_or_path,
                )

    def retokenize(self, sentence_features: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids = sentence_features["input_ids"]
        device = input_ids.device
        sentences_decoded = self.tokenizer_encoder.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        retokenized = self.tokenizer_decoder(
            sentences_decoded, padding=True, truncation="longest_first", return_tensors="pt", max_length=None
        ).to(device)
        return retokenized

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        source_features, target_features = tuple(sentence_features)
        if self.need_retokenization:
            # since the sentence_features here are all tokenized by encoder's tokenizer,
            # retokenization by the decoder's one is needed if different tokenizers used
            target_features = self.retokenize(target_features)
        reps = self.encoder(source_features)["sentence_embedding"]  # (bsz, hdim)

        # Prepare input and output
        target_length = target_features["input_ids"].shape[1]
        decoder_input_ids = target_features["input_ids"].clone()[:, : target_length - 1]
        label_ids = target_features["input_ids"][:, 1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=None,
            attention_mask=None,
            encoder_hidden_states=reps[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
            encoder_attention_mask=source_features["attention_mask"][:, 0:1],
            labels=None,
            return_dict=None,
            use_cache=False,
        )

        # Calculate loss
        lm_logits = decoder_outputs[0]
        ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
        if hasattr(self, "current_epoch_loss"):
            self.current_epoch_loss += loss.item()

        return loss


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
    for i in range(len(X) - 1):
        train_examples.append(InputExample(texts=[X[i],X[i + 1]], label=y[i]))
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
            plt.title('TSNE Visualization of Train Embeddings-DenoisingAutoEncoderLoss-loss')
            plt.savefig(f'plots/{data}_DenoisingAutoEncoderLoss_TSNE_train' + str(self.epoch) + '.png')


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
            model.save(os.path.join(self.save_path, f"DenoisingAutoEncoderLoss_{model_name}_{data}"))

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

custom_loss_fn = DenoisingAutoEncoderLoss(model=model)  # Replace original loss initialization

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

with open(f"metrics/{data}_DenoisingAutoEncoderLoss_val.npy", 'wb') as g2:
   np.save(g2, val_accuracies)
   np.save(g2, val_f1_scores)

with open(f"metrics/{data}_DenoisingAutoEncoderLoss_test.npy", 'wb') as g3:
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
   
plot_metrics("Accuracy",  val_accuracies, test_accuracies, "Accuracy Over Epochs",ep,save_plot="plots/"+data+"_"+model_name+"_DenoisingAutoEncoderLosss_accuracy.png")
plot_metrics("F1 Score",  val_f1_scores, test_f1_scores, "F1 Score Over Epochs",ep,save_plot="plots/"+data+"_"+model_name+"_DenoisingAutoEncoderLoss_f1scores.png")

# After training is complete, use the accumulated_losses_per_epoch list from the CustomLoss instance
epochs = list(range(1, len(model.training_losses) + 1))
epoch_losses = model.training_losses

np.save(f'embeddings/{data}_{model_name}_train_DenoisingAutoEncoderLoss.npy',epoch_losses)
# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(epochs, epoch_losses, linestyle='-', color='b')
plt.title(' DenoisingAutoEncoderLoss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.xticks(epochs)  # Ensure x-axis has ticks for each epoch
plt.tight_layout()
plt.savefig("plots/"+data+"_"+model_name+"_DenoisingAutoEncoderLoss_losses.png", format='png', bbox_inches='tight', dpi=300)
