import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging 
import numpy as np 
import joblib 
import hdbscan 
from mlp_utils import * 
import sys
sys.path.insert(1, '/sc/arion/projects/tauomics/danielk/efficient-kan/src')
from efficient_kan import KAN as eKAN 

def setup_logging():
    logger = logging.getLogger('KAN_ModelProcessingLogger')
    logger.setLevel(logging.INFO)  

    fh = logging.FileHandler('hemosiderin/MLPtrainer/minerva_out/kan_logfile.log')  
    ch = logging.StreamHandler()  

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

logger = setup_logging()
logger.info('Logger initialized...')

df = pd.read_pickle('/sc/arion/projects/tauomics/danielk/full_results_prcntl_df.pkl.gzip', compression='gzip')
logger.info('Large DF loaded into memory...')

feat_dir = '/sc/arion/projects/tauomics/FeatureVectors/ParkmanRHI/dino/'
def get_feats(pairs): 
    slide_mapper = {}
    out_fts = np.empty((len(pairs), 384))
    for i, (sid, coord) in enumerate(tqdm(pairs)): 
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

def process_batch(batch_df, model):
    features = get_feats(batch_df[["Slide_ID", "Coords"]].to_numpy())
    
    features_tensor = torch.tensor(features).float().to(device)
    with torch.no_grad():
        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)

    batch_df['logits'] = logits.cpu().numpy().tolist()
    batch_df['softmax'] = probs.cpu().numpy().tolist()
    batch_df['probability'] = probs.max(dim=1).values.cpu().numpy().tolist()
    batch_df['predicted_label'] = predictions.cpu().numpy().tolist()
    return batch_df

def process_in_batches(df, model, batch_size=64):
    num_batches = (len(df) + batch_size - 1) // batch_size  
    processed_batches = []

    print_every = int(np.round(num_batches / 100))
    for i in tqdm(range(num_batches)):
        if i % print_every == 0: logger.info(f'{i} of {num_batches}')
        batch_df = df[i * batch_size:(i + 1) * batch_size]
        processed_batch = process_batch(batch_df, model)
        processed_batches.append(processed_batch)

    return pd.concat(processed_batches, ignore_index=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = joblib.load('/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/kan_saves/0_model.pkl')

model = eKAN(layers_hidden=[384,256,128,3], grid_size=5)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
logger.info('Model loaded.')

processed_df = process_in_batches(df, model, batch_size=256)
logger.info('Dataframe constructed.')

processed_df.to_pickle('/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/kan_outputs/kan_processed_data.pkl.gzip', compression='gzip')
