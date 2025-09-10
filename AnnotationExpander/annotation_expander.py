# import os
# from os import listdir 
from os.path import join 
# from pandas import read_pickle, DataFrame, read_csv
import numpy as np 
# import pandas as pd 
import openslide 
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
from sklearn import svm
from sklearn.cluster import KMeans
import torch 
# from sklearn.semi_supervised import LabelSpreading
import umap.parametric_umap as pumap
import hdbscan
# import random
import seaborn as sns 
# from collections import defaultdict
import joblib 
# from PIL import Image
# import torchvision.transforms as transforms

import sys 
# sys.path.insert(1, '/sc/arion/projects/tauomics/danielk/hover_net_models/dino')
# import vision_transformer as vits 
# import json 
from functools import reduce

def save_data(X, y, new_indices, new_labels, unused_feats, unused_pairs, unused_indices, round_name, notes): 
    ulr = {
        'X': np.concatenate([X, unused_feats[new_indices]]), 
        'y': np.concatenate([y, new_labels]), 
        'slide_coord': unused['slide_coords'][indices_negative], 
        'unused_pairs': unused_pairs[unused_indices], 
        'unused_feats': unused_feats[unused_indices], 
        'unused_indices': unused_indices, 
        'round_name': round_name, 
        'new_indices': new_indices, 
        'notes': notes
    }
    return ulr, joblib.dump(ulr, f'/sc/arion/projects/tauomics/danielk/hemosiderin/hemosiderin_round_{round_name}_added.pkl')

def load_data(round_name): 
    return joblib.load(f'/sc/arion/projects/tauomics/danielk/hemosiderin/hemosiderin_round_{round_name}_added.pkl')

image_path = '/sc/arion/projects/tauomics/ParkmanRHI'
path_for_slide = lambda x: join(image_path, x + '.svs')
def get_tile(slide, coords): 
    if slide == '0_5ad0c3f8-0fdf-1d64-b8e2-feae32f3224c_122233': 
        slide = '5ad0c3f8-0fdf-1d64-b8e2-feae32f3224c_122233'
    x, y = np.array(coords).astype(int)
    wsi = openslide.open_slide(path_for_slide(slide))
    return wsi.read_region((x, y), 0, (256, 256))

# get feats 
feat_dir = '/sc/arion/projects/tauomics/FeatureVectors/ParkmanRHI/dino/'
def get_feats(pairs): 
    slide_mapper = {}
    out_fts = np.empty((len(pairs), 384))
    for i, (sid, coord) in enumerate(pairs): 
        if sid not in slide_mapper.keys(): 
            with torch.no_grad(): 
                fts = torch.load(join(feat_dir, 'features', sid + '.pt')).detach().cpu().numpy()
                crds = torch.load(join(feat_dir, 'coords', sid + '.pt')).detach().cpu().numpy()
                coord_dict = {tuple(crd): idx for idx, crd in enumerate(crds)}
                slide_mapper[sid] = {'feats': fts, 'coords': coord_dict}
                
        fts = slide_mapper[sid]['feats']
        coord_dict = slide_mapper[sid]['coords']
        coord_tuple = tuple(map(int, coord))
        j = coord_dict[coord_tuple]
        out_fts[i] = fts[j]
    return out_fts 

def display_images(slide_coords, n_start=0, n_images=64, nrows=8, ncols=8):
    random_order = np.random.choice(len(slide_coords), min(n_images, len(slide_coords)), replace=False)
    end_index = min(n_start + n_images, len(slide_coords))
    selected_images = slide_coords[random_order][n_start:end_index].squeeze()
    selected_images = [get_tile(*_) for _ in selected_images]
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2, nrows*2))
    axes = axes.flatten()
    
    for ax, img in zip(axes, selected_images):
        ax.imshow(img)  
        ax.axis('off')  

    for ax in axes[len(selected_images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

class AnnotationExpander:
    def __init__(self):
        self.X = None
        self.y = None
        self.unused_pairs = None 
        self.unused_feats = None 
        self.round_num = None
        self.umap = None 
        self.embedding = None 
        self.unused_embeddings = None 
        self.clusterer = None 
        self.cluster_labels = None 
        self.temp_feat_holder = None 
    
    def set_label_for_indices(self, indices, new_labels):
        new_features = self.unused_feats[indices]
        if self.temp_feat_holder: 
            new_features = self.temp_feat_holder[indices]
        
        len_before = len(self.y)
        self.X = np.concatenate([self.X, new_features])
        self.y = np.concatenate([self.y, new_labels])
        
        len_unlabel_before = len(self.unused_pairs)
        self.unused_feats = np.delete(self.unused_feats, indices, axis=0)
        self.unused_pairs = np.delete(self.unused_pairs, indices, axis=0)
        
        assert len(self.y) - len_before == len_unlabel_before - len(self.unused_pairs)
        self.save_state()
    
    def save_state(self):
        ulr = {
            'X': self.X,
            'y': self.y,
            'unused_pairs': self.unused_pairs,
            'unused_feats': self.unused_feats, 
            'clusterer': self.clusterer, 
            'cluster_labels': self.cluster_labels, 
            'embedding': self.embedding, 
            'unused_embedding': self.unused_embeddings 
        }
        if self.temp_feat_holder: 
            ulr['unused_feats'] = self.temp_feat_holder 
            ulr['subbed_feats'] = self.unused_feats 
            
        round_name = str(self.round_num+1)
        joblib.dump(ulr, f'/sc/arion/projects/tauomics/danielk/hemosiderin/hemosiderin_round_{round_name}_added.pkl')
        
    def load_state(self, round_name):
        filepath = f'/sc/arion/projects/tauomics/danielk/hemosiderin/hemosiderin_round_{round_name}_added.pkl'
        data = joblib.load(filepath)
        self.X = data['X']
        self.y = data['y']
        self.unused_pairs = data['unused_pairs'].squeeze()
        self.unused_feats = data['unused_feats'].squeeze()
        self.round_num = int(round_name)
        
    def train_umap(self, n_components=2): 
        model = pumap.ParametricUMAP(n_components=n_components, 
                             n_neighbors=15,
                             min_dist=0.1,
                             target_metric='categorical')

        embedding = model.fit_transform(self.X, y=self.y)
        self.umap = model 
        self.embedding = embedding 
        self.unused_embeddings = model.transform(self.unused_feats)
        
    def train_hdbscan(self, min_cluster_size=50, min_samples=50, gen_min_span_tree=True): 
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                    min_samples=min_samples, gen_min_span_tree=gen_min_span_tree)
        self.cluster_labels = clusterer.fit_predict(self.unused_embeddings) 
        self.clusterer = clusterer 
        
    def save_umap(self, round_name): 
        if self.umap: 
            if not exists(f'/sc/arion/projects/tauomics/danielk/hemosiderin/round_{round_name}_umap'): 
                umap.save(f'/sc/arion/projects/tauomics/danielk/hemosiderin/round_{round_name}_umap')
            else: 
                print('Round name corresponds to already saved umap...')
        else: 
            print('Umap not initialized...')
            
    def use_image_reduced_feats(self, new_feats): 
        self.temp_feat_holder = self.unused_feats 
        self.unused_feats = new_feats 
        
    def remove_image_reduced_feats(self): 
        self.unused_feats = self.temp_feat_holder
        self.temp_feat_holder = None 
            
    def visualize_embedding(self, dim_1=0, dim_2=1): 
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        scatter = ax[0].scatter(self.unused_embeddings[:, dim_1], self.unused_embeddings[:, dim_2], 
                              c=self.cluster_labels, s=5, cmap='Spectral', alpha=0.1)
        ax[1].scatter(self.embedding[:, dim_1], self.embedding[:, dim_2], s=1, c=self.y, cmap='Spectral', 
                    linewidths=5, marker='x', alpha=.06, edgecolors='black')
        plt.colorbar(scatter, label='Y label')
        plt.show()
        
    def visualize_clusters(self): 
        for c, l in zip(*np.unique(self.cluster_labels, return_counts=True)): 
            print(c, l)
            display_images(self.unused_pairs[(self.cluster_labels==c)], n_start=0, n_images=10, ncols=10, nrows=1)
            
    def visualize_single_cluster(self, label, n=64, ncols=8, nrows=8): 
        display_images(self.unused_pairs[self.cluster_labels==label], n_start=0, n_images=n, ncols=ncols, nrows=nrows)
        
    def save_from_label_mapper(self, label_mapper): 
        label_boolean = {}
        for label, cls in label_mapper.items(): 
            label_boolean[label] = reduce(np.logical_or, [self.cluster_labels==i for i in cls])
        
        indices = np.where(reduce(np.logical_or, [_ for _ in label_boolean.values()]))[0]
        not_indices = np.where(~reduce(np.logical_or, [_ for _ in label_boolean.values()]))[0]
        labels = np.concatenate([[l]*np.sum(logic_array) for l, logic_array in label_boolean.items()])
        assert (len(indices) + len(not_indices)) == len(self.unused_pairs)
        self.set_label_for_indices(indices, labels)
        self.save_state()