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

from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.utils.inference import CustomKNN

import logging

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import umap
from cycler import cycler
from PIL import Image


import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import datetime

import argparse 

parser = argparse.ArgumentParser(description='Generate and Save Hemosiderin MLP')
parser.add_argument('--pca', action='store_true')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--out_dim', type=int, default=128)
parser.add_argument('--trunk_lr', type=float, default=0.001)
parser.add_argument('--classifier_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=4)
parser.add_argument('--pca_ncomponents', type=int, default=128)
parser.add_argument('--metric_loss_w', type=float, default=0.75)
parser.add_argument('--classifier_loss_w', type=float, default=0.25)
args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)

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

if __name__ == "__main__": 
    input_dim = 384
    if args.pca: input_dim = args.pca_ncomponents
    output_dim = args.out_dim

    if args.model == 'MLP_ATTN': 
        mlp = MLPWithAttention(input_dim, output_dim).to(device)
    if args.model == 'MLP': 
        mlp = MLP(input_dim, output_dim)

    ulr = joblib.load('/sc/arion/projects/tauomics/danielk/hemosiderin/hemosiderin_round_7_added.pkl')
    
    saver = {'args': args}
    
    X = ulr['X']
    y = ulr['y']

    print(np.unique(y, return_counts=True))

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
    saver['models'] = models

    optimizers = {
        "trunk_optimizer": trunk_optimizer, 
        'classifier_optimizer': classifier_optimizer
    }

    loss_funcs = {"metric_loss": loss_lsl, "classifier_loss": loss_ce}
    mining_funcs = {"tuple_miner": miner}

    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": args.metric_loss_w, "classifier_loss": args.classifier_loss_w}

    record_keeper, _, _ = logging_presets.get_record_keeper(
        "example_logs", "example_tensorboard"
    )
    hooks = logging_presets.get_hook_container(record_keeper)
    dataset_dict = {"test": test_dataset}
    model_folder = "example_saved_models"

    # def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    #     logging.info("UMAP plot for the {} split and label set {}".format(split_name, keyname))
    #     label_set = np.unique(labels)
    #     num_classes = len(label_set)
    #     plt.figure(figsize=(20, 15))
    #     plt.gca().set_prop_cycle(
    #         cycler(
    #             "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
    #         )
    #     )
    #     for i in range(num_classes):
    #         idx = labels == label_set[i]
    #         plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)

    #     # Save the plot
    #     plt.savefig(f"umap_{split_name}_{keyname}.png", dpi=300)


    # knn_func = CustomKNN(SNRDistance())
    # custom_acc = AccuracyCalculator(include=(),
    #                     exclude=(),
    #                     avg_of_avgs=True,
    #                     return_per_class=False,
    #                     k="max_bin_count",
    #                     label_comparison_fn=None,
    #                     device=None,
    #                     knn_func=knn_func,
    #                     kmeans_func=None)



    # # Create the tester
    # tester = testers.GlobalEmbeddingSpaceTester(
    # # tester = testers.BaseTester(
    #     use_trunk_output=True,
    #     batch_size=1, 
    #     end_of_testing_hook=hooks.end_of_testing_hook,
    #     visualizer=umap.UMAP(),
    #     visualizer_hook=visualizer_hook,
    #     dataloader_num_workers=0,
    #     accuracy_calculator=AccuracyCalculator(k="max_bin_count")
    #     # accuracy_calculator=custom_acc,
    #     # accuracy_calculator=AccuracyCalculator(exclude=("NMI", "AMI"), 
    #     #             knn_func=CustomKNN(SNRDistance())),
    # )

    # end_of_epoch_hook = hooks.end_of_epoch_hook(
    #     tester, dataset_dict, model_folder, test_interval=1, patience=5
    # )

    # trainer = trainers.TrainWithClassifier(
    #     models,
    #     optimizers,
    #     batch_size,
    #     loss_funcs,
    #     train_dataset,
    #     mining_funcs=mining_funcs,
    #     sampler=sampler,
    #     dataloader_num_workers=0,
    #     loss_weights=loss_weights,
    #     end_of_iteration_hook=hooks.end_of_iteration_hook,
    #     end_of_epoch_hook=end_of_epoch_hook,
    # )

    trainer.train(num_epochs=num_epochs)

    # name_key = str(np.random.randint(100000, 1000000)) + 'output.pkl'
    name_key = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S") + '_output.pkl'
    
    saver['train'] = {'X': X_train.detach().cpu(), 'y': y_train.detach().cpu()}
    saver['test'] = {'X': X_test.detach().cpu(), 'y': y_test.detach().cpu()}

    joblib.dump(saver, '/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/saves/'+name_key)