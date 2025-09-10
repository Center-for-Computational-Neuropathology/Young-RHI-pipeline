import numpy as np 
import pandas as pd 
import joblib 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm.auto import tqdm 
from sklearn.model_selection import train_test_split
from os.path import join 
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.decomposition import PCA 
from kan import KAN 
from torch.nn import Softmax, CrossEntropyLoss
from torch import Tensor, cuda 
from torch import device as d 
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt 

device = d('cuda:0' if cuda.is_available() else 'cpu')
print(device)

rhi = pd.read_csv('/sc/arion/projects/tauomics/danielk/clinical_data/parkman_rhi_tsvs/parkman_rhi_clinical_full.tsv', sep='\t')
rhi['Slide_ID'] = rhi['Slide_ID'].str.replace('.svs', '', regex=False)

slide_ids = joblib.load('/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/outputs/umap_clustering_interpretation_of_dino/bag_slide_ids.pkl')

cluster_save_data = joblib.load('/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/outputs/umap_clustering_interpretation_of_dino/clusters_pred=0,1.pkl')
cluster_labels = cluster_save_data['labels']
n_clusters = len(np.unique(cluster_labels))


slide_decomposition_mapper = joblib.load('/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/outputs/umap_clustering_interpretation_of_dino/cluster_decomps_pred=0,1.pkl')
decompositions = np.zeros((len(slide_ids), n_clusters*384*2))

for i, slide_id in enumerate(tqdm(slide_ids)): 
    decompositions[i] = slide_decomposition_mapper[slide_id]

rhi['cte'] = (rhi['CTE'] == 'Yes').astype(int)

rhi['Brief TBRI'] = (rhi['tbri'] >= 65).astype(int)
rhi['Brief TMI'] = (rhi['tmi'] >= 65).astype(int)
rhi['AES'] = (rhi['aestot'] >= 34).astype(int)
rhi['GDS'] = (rhi['GDStot'] >= 5).astype(int)
rhi['BIS'] = (rhi['bistot'] >= 80).astype(int)

def under_30_cte(row): 
    if row['agedeath'] > 30: 
        return np.nan 
        # return 0
    else: 
        return row['cte']

rhi['under_30_cte'] = rhi.apply(under_30_cte, axis=1)

def prg_gain_transform(x, pi): 
    if x == pi == 1: 
        return 1 
    return (x - pi) / ((1 - pi) * x)

def f1_gain_score(labels, preds): 
    pi = np.sum(labels) / len(labels)
    return prg_gain_transform(f1_score(labels, preds), pi)

def get_stats(lbls, ps): 
    auc = roc_auc_score(lbls, ps)
    fpr, tpr, thresholds = roc_curve(lbls, ps)
    youdens_j = tpr - fpr
    max_index = np.argmax(youdens_j)
    optimal_threshold = thresholds[max_index]
    max_youdens_j = youdens_j[max_index]
    preds = (ps >= optimal_threshold).astype(int)
    f1 = f1_gain_score(lbls, preds)
    return auc, f1 

def slide_to_label(df, column): 
    sub_df = df[['Slide_ID', column]].dropna()
    return dict(zip(sub_df['Slide_ID'].to_numpy(), sub_df[column].to_numpy()))

def run_kan(column, clusters=[0, 1, 2, 3, 4], steps=5, max_k=20): 
    #trying with only certain clusters 
    cluster_to_try = clusters
    indices = np.concatenate([range((c+1)*384*2, (c+2)*384*2) for c in cluster_to_try])
    sub_decomposition = decompositions[:, indices]

    sub_rhi = rhi[['Slide_ID', column]].dropna()
    slide_label_map = slide_to_label(sub_rhi, column)
    ls = np.array([slide_label_map[s] for s in slide_ids if s in slide_label_map])

    inner_slides = np.array([i for i, s in enumerate(slide_ids) if s in slide_label_map])
    sub_decompositions = sub_decomposition[inner_slides,:]

    out_df = []
    best_model = None
    best_auc = 0 
    pca = PCA(n_components=100)
    sub_decompositions = pca.fit_transform(sub_decompositions)
    previous_ls = None 
    states = np.random.randint(low=0, high=1000000, size=max_k)
    for k in range(0, max_k): 
        print(k, 'of', max_k)
        X_train, X_test, y_train, y_test = train_test_split(sub_decompositions, range(len(ls)), test_size=0.3, 
                                                            stratify=ls, random_state=states[k])

        dataset = {}
        dataset['train_input'] = Tensor(X_train).to(device)
        dataset['train_label'] = Tensor(ls[y_train]).long().to(device)
        dataset['test_input'] = Tensor(X_test).to(device)
        dataset['test_label'] = Tensor(ls[y_test]).long().to(device)
        
        assert not np.array_equal(previous_ls, ls[y_test])
        previous_ls = ls[y_test]
        
        model = KAN(width=[sub_decompositions.shape[1], 16, 2], device=device)
        results = model.train(dataset, steps=steps, loss_fn=CrossEntropyLoss(), device=device)
        m = Softmax(dim=1)
        probs = m(model(dataset['test_input']))
        preds = probs.argmax(dim=1)
        probs = probs[:, 1]

        auc, f1 = get_stats(ls[y_test], probs.cpu().data.numpy())
        print(auc, f1)

        if auc > best_auc: 
            best_auc = auc 
            best_model = model 
        out_df.append(
            {
                'k': k, 
                'AUC': auc, 
                'F1': f1
            }
        )

    
    out_df = pd.DataFrame(out_df)
    print(np.mean(out_df['AUC']))
    print(np.mean(out_df['F1']))
    stat, p_value = wilcoxon(out_df['AUC'] - 0.5)
    print(f'Statistics={stat}, p={p_value}')

    melted_df = pd.melt(out_df, id_vars=['k'], value_vars=['AUC', 'F1'], var_name='Metric', value_name='Score')

    fig, ax = plt.subplots(1, 1, figsize=(3, 8))
    sns.boxplot(data=melted_df, x='Metric', y='Score', color='white')
    sns.stripplot(melted_df, y='Score', x='Metric', color='black')
    plt.tight_layout()
    plt.savefig('/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/kan_outputs/u30cte_performance.jpg')
    plt.show()
    
    # lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    # best_model.auto_symbolic(lib=lib)
    # print(best_model.symbolic_formula())
    return out_df, p_value, best_model 

print('lets go')
u30_df, u30_p, u30_model = run_kan('under_30_cte', steps=10, max_k=20)
print('noice')

# u30_model.plot()
# plt.savefig('/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/kan_outputs/u30cte_model.jpg')
# print('heard')

out_data = {
    'df': u30_df.to_numpy(), 
    'p': u30_p, 
    'model': u30_model.state_dict()
}
joblib.dump(out_data, '/sc/arion/projects/tauomics/danielk/hemosiderin/MLPtrainer/kan_outputs/u30cte_ckpt.pkl')
print('peace')