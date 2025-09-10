from os.path import join 
import torch 
import numpy as np 
import joblib 
import tensorflow as tf 
from tqdm import tqdm 

def npify(iterable_thing): 
    if isinstance(iterable_thing, np.ndarray): 
        return iterable_thing
    elif torch.is_tensor(iterable_thing): 
        return iterable_thing.detach().cpu().numpy()
    else: 
        return np.array(iterable_thing)

def stringifyCoord(coord):
    x, y = np.array(coord).astype(int)
    return f'{x}_{y}'

def destringifyCoord(hashstring): 
    return np.array(hashstring.split('_')).astype(int)

def stringifySlideAndCoord(slide_id, coord): 
    return str(slide_id) + '_c_' + stringifyCoord(coord)

def destringifySlideAndCoord(hashstring): 
    slide_id = hashstring.split('_c_')[0]
    x, y = destringifyCoord(hashstring.split('_c_')[1])
    return slide_id, x, y 

def getFeaturesAsMap(slide_id, directory): 
    with torch.no_grad(): 
        feats = torch.load(join(directory, 'features', slide_id + '.pt')).detach().cpu().numpy()
        coords = torch.load(join(directory, 'coords', slide_id + '.pt')).detach().cpu().numpy()
        out_dict = {stringifyCoord(coord): feats[i] for i, coord in enumerate(coords)}
    return out_dict 

def resaveFeatsAndCoords(slide_id, directory): 
    out_dict = getFeaturesAsMap(slide_id, directory)
    joblib.dump(out_dict, join(directory, 'hashmaps', slide_id + '.pkl'), compress='gzip')

def getFeature(slide_id, coord, directory): 
    hash_string = stringifyCoord(coord)
    hash_map = joblib.load(join(directory, 'hashmaps', slide_id + '.pkl'))
    return hash_map[hash_string]

def getFeatureSaveMap(slide_id, coord, directory, map_of_maps): 
    hash_string = stringifyCoord(coord)
    if not slide_id in map_of_maps.keys(): 
        map_of_maps[slide_id] = joblib.load(join(directory, 'hashmaps', slide_id + '.pkl'))
    return map_of_maps[slide_id][hash_string], map_of_maps

def getFeatureSaveMapOld(slide_id, coord, directory, map_of_maps): 
    hash_string = stringifyCoord(coord)
    if not slide_id in map_of_maps.keys(): 
        map_of_maps[slide_id] = getFeaturesAsMap(slide_id, directory)
    return map_of_maps[slide_id][hash_string], map_of_maps

def batch_process(fvs, pumap_model, umap_embedding_size=2, batch_size=512): 
    embeddings = np.empty((len(fvs), umap_embedding_size))

    dataset = tf.data.Dataset.from_tensor_slices(fvs) 
    dataset = dataset.batch(batch_size)

    for i, batch in tqdm(enumerate(dataset)): 
        batch_embeddings = pumap_model.transform(batch)
        embeddings[i*batch_size:(i+1)*batch_size] = batch_embeddings

    return embeddings