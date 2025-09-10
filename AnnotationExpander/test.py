import pandas as pd 
from utils import * 
from os import listdir
from os.path import join 
import os 
import gc 
import umap.parametric_umap as umap 
from umap.parametric_umap import load_ParametricUMAP
import joblib 
import hdbscan 
from tqdm import tqdm 

import resource 

max_memory_mb = int(os.environ['LSB_SUB_RES_REQ'].split('mem=')[1].split(']')[0])-500 # subtract 500 for safety 
max_memory_bytes = max_memory_mb * 1024 * 1024 
resource.setrlimit(resource.RLIMIT_RSS, (max_memory_bytes, max_memory_bytes))

import argparse 

parser = argparse.ArgumentParser(description='Resave as hashmaps')
parser.add_argument('--feat_dir', type=str, default=None)
args = parser.parse_args()

# START 
print('Starting analysis')

# load dataframe 
df = pd.read_pickle('/sc/arion/projects/tauomics/danielk/regioned_results_prcntl_df.pkl.gzip', compression='gzip')
print('Read DF')

# load parametric umap 
model = load_ParametricUMAP('/sc/arion/projects/tauomics/danielk/hemosiderin/hemosiderin_umap_round_2')
print('Loaded parametric UMAP') 

# do computations 
with torch.no_grad(): 
    total = len(df)

    feats = np.empty((total, 384))
    slide_ids = np.empty((total)).astype(str)
    coords = np.empty((total, 2))

    map_holder = {} # stores the slide hashmaps 

    for i, (slide_id, coord) in tqdm(enumerate(df[['Slide_ID', 'Coords']].to_numpy())): 
        featurevector, _ = getFeatureSaveMapOld(slide_id, coord, args.feat_dir, map_holder)

        feats[i] = featurevector 
        slide_ids[i] = slide_id
        coords[i] = coord
    print('Loaded FVs')

    del map_holder 
    gc.collect() # get outta here 
    print('Made space')

    embeddings = batch_process(feats, model, umap_embedding_size=model.n_components, batch_size=512)
    print('Embedded FVs')

    del feats 
    gc.collect() 
    print('Made space')

    joblib.dump(embeddings, '/sc/arion/projects/tauomics/danielk/hemosiderin/massive_pmap_embedding.pkl', compress='gzip')
    print('Got and saved embedding', embeddings.shape)

    save_dict = {
        'slides': slide_ids, 
        'coords': coords, 
        'embedding': embeddings 
    }
    joblib.dump(save_dict, '/sc/arion/projects/tauomics/danielk/hemosiderin/massive_save_dict.pkl')
    print('Saved embeddings dict')

    clusterer = joblib.load('/sc/arion/projects/tauomics/danielk/hemosiderin/hemosiderin_hdbscan_round_2.pkl')
    clusterer.generate_prediction_data()
    labels, strengths = hdbscan.approximate_predict(clusterer, embeddings)

    joblib.dump(labels, '/sc/arion/projects/tauomics/danielk/hemosiderin/massive_hdbscan_labels.pkl', compress='gzip')
    print('Saved clustered labels')