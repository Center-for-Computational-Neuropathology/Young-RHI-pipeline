import torch
import torch.nn as nn
from torch.nn import BCELoss
import torch.nn.functional as F

import joblib
import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from pytorch_metric_learning import losses, miners, samplers, testers, trainers

import logging

import matplotlib.pyplot as plt

import datetime

import argparse 

parser = argparse.ArgumentParser(description='Generate and Save Hemosiderin MLP')
parser.add_argument('--pca', action='store_true')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--out_dim', type=int, default=128)
parser.add_argument('--job_n', type=str, default='11')
parser.add_argument('--trunk_lr', type=float, default=0.001)
parser.add_argument('--classifier_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=4)
parser.add_argument('--pca_ncomponents', type=int, default=128)
parser.add_argument('--metric_loss_w', type=float, default=0.75)
parser.add_argument('--classifier_loss_w', type=float, default=0.25)
args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        # Apply softmax to attention weights
        weights = F.softmax(self.attention_weights, dim=0)
        # Multiply input by attention weights
        x = x * weights
        return x

class MLPWithAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPWithAttention, self).__init__()
        self.attention = AttentionLayer(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.targets = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sample.to(device), target.to(device)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), 'checkpoint.pt')
    #     self.val_loss_min = val_loss

def calculate_snr(embeddings, labels):
    unique_labels = torch.unique(labels)
    class_means = []
    within_class_vars = []

    for label in unique_labels:
        class_embeddings = embeddings[labels == label]
        class_mean = torch.mean(class_embeddings, dim=0)
        class_means.append(class_mean)
        within_class_vars.append(torch.var(class_embeddings, dim=0))

    class_means = torch.stack(class_means)
    overall_mean = torch.mean(class_means, dim=0)
    between_class_var = torch.mean((class_means - overall_mean) ** 2)
    within_class_var = torch.mean(torch.stack(within_class_vars))

    snr = between_class_var / (within_class_var + 1e-6)  # Adding a small constant to avoid division by zero
    return snr

if __name__ == "__main__": 
    input_dim = 384
    if args.pca: input_dim = args.pca_ncomponents
    output_dim = args.out_dim

    if args.model == 'MLP_ATTN': 
        mlp = MLPWithAttention(input_dim, output_dim).to(device)
    if args.model == 'MLP': 
        mlp = MLP([input_dim, output_dim])

    job_n = args.job_n 
    ulr = joblib.load(f'/sc/arion/projects/tauomics/danielk/hemosiderin/hemosiderin_round_{job_n}_added.pkl')
    
    saver = {'args': args}
    
    X = ulr['X']
    y = ulr['y']

    logging.info(np.unique(y, return_counts=True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)
    
    label_encoder = LabelEncoder()
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = label_encoder.fit_transform(torch.tensor(y_train, dtype=torch.long))
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = label_encoder.transform(torch.tensor(y_test, dtype=torch.long))

    if args.pca: 
        pca = PCA(n_components=args.pca_ncomponents)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        saver['pca'] = pca

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    loss_lsl = losses.LiftedStructureLoss(neg_margin=1, pos_margin=0)
    loss_ce = torch.nn.CrossEntropyLoss()

    # Set the classification loss:
    classification_loss = torch.nn.CrossEntropyLoss()

    # Set the mining function
    miner = miners.HDCMiner(filter_percentage=0.5)

    # Set the dataloader sampler
    sampler = samplers.MPerClassSampler(
        train_dataset.targets, m=128, length_before_new_iter=len(train_dataset)
    )

    classifier = MLP([output_dim, 3]).to(device)

    trunk_optimizer = torch.optim.Adam(mlp.parameters(), lr=args.trunk_lr, weight_decay=0.0001)

    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=args.classifier_lr, weight_decay=0.001
    )

    # Set other training parameters
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    # Package the above stuff into dictionaries.
    models = {"trunk": mlp, 'classifier': classifier}
    
    # Initialize logging dataframe
    logs = pd.DataFrame(columns=["Epoch", "Batch", "Train Loss", 
        "Validation Loss", "Train Accuracy", "Validation Accuracy", 
        "Class Accuracy", "F1 Score", "SNR Distance"]
    )

    # Instantiate early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)

    mlp.to(device)
    classifier.to(device)
    for epoch in range(num_epochs): 
        mlp.train()
        classifier.train()

        total_loss = 0
        train_accuracy = 0 
        for batch_idx, (data, targets) in enumerate(DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)):
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            embeddings = mlp(data)
            output = classifier(embeddings)

            # Calculate losses
            hard_pairs = miner(embeddings, targets)
            metric_loss = loss_lsl(embeddings, targets, hard_pairs)
            classifier_loss = loss_ce(output, targets)

            # Weighted loss
            loss = args.metric_loss_w * metric_loss + args.classifier_loss_w * classifier_loss
            total_loss += loss.item()

            # Backward and optimize
            trunk_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            loss.backward()
            trunk_optimizer.step()
            classifier_optimizer.step()

            preds = output.argmax(dim=1)
            train_accuracy += (preds == targets).sum().item()

             # Logging
            if batch_idx % 100 == 0:
                logging.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}')

        # Evaluate
        mlp.eval()
        classifier.eval()
        epoch_embeddings = []
        epoch_labels = []
        class_correct = dict()
        class_total = dict()
        with torch.no_grad():
            total_accuracy = 0
            total_f1 = 0
            batch_count = 0
            validation_loss = 0 
            for data, targets in DataLoader(test_dataset, batch_size=args.batch_size):
                data, targets = data.to(device), targets.to(device)
                embeddings = mlp(data)
                output = classifier(embeddings)

                epoch_embeddings.append(embeddings.detach().cpu())
                epoch_labels.append(targets.detach().cpu())

                preds = output.argmax(dim=1)
                correct = preds == targets 
                total_accuracy += (preds == targets).sum().item()
                total_f1 += f1_score(targets.cpu(), preds.cpu(), average='macro')
                batch_count += 1

                for label, is_correct in zip(targets.cpu().numpy(), correct.cpu().numpy()):
                    if label in class_correct:
                        class_correct[label] += int(is_correct)
                        class_total[label] += 1
                    else:
                        class_correct[label] = int(is_correct)
                        class_total[label] = 1

                # hard_pairs = miner(embeddings, targets)
                # metric_loss = loss_lsl(embeddings, targets, hard_pairs)
                metric_loss = loss_lsl(embeddings, targets)
                classifier_loss = loss_ce(output, targets)
                loss = args.metric_loss_w * metric_loss + args.classifier_loss_w * classifier_loss
                validation_loss += loss.item()
            
            # SNR distance 
            epoch_embeddings = torch.cat(epoch_embeddings, dim=0)
            epoch_labels = torch.cat(epoch_labels, dim=0)
            current_snr_distance = calculate_snr(epoch_embeddings, epoch_labels).item()

            accuracy = total_accuracy / len(test_dataset)
            class_accuracies = {cls: class_correct[cls] / class_total[cls] for cls in class_correct}
            f1 = total_f1 / batch_count
            average_epoch_loss = total_loss / len(DataLoader(train_dataset, batch_size=batch_size, sampler=sampler))
            average_validation_loss = validation_loss / len(DataLoader(test_dataset, batch_size=args.batch_size))
            train_accuracy = train_accuracy / len(train_dataset)
            logs.loc[len(logs)] = [epoch, batch_idx, average_epoch_loss, average_validation_loss, train_accuracy, accuracy, class_accuracies, f1, current_snr_distance]  
        
        # Early stopping check
        early_stopping(validation_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

    name_key = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S") 
    name_key = name_key if not args.pca else name_key + '_dim_reduce' 
    name_key = name_key + '_' + args.model 
    name_key = name_key + '_output.pkl'

    saver['models'] = models
    saver['train'] = {'X': X_train, 'y': y_train}
    saver['test'] = {'X': X_test, 'y': y_test}
    saver['trunk_state_dict'] = mlp.state_dict()
    saver['classifier_state_dict'] = classifier.state_dict()
    saver['logs'] = logs
    joblib.dump(saver, '/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/saves/'+name_key)