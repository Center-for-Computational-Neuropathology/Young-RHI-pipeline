# Deep Learning Pipeline for Characterizing Early Microstructural Brain Changes Following Repetitive Head Impacts

## Overview

This repository contains a novel deep learning pipeline for analyzing whole slide images (WSIs) of human brain tissue to characterize the earliest microstructural changes following repetitive head impacts (RHI) that may lead to chronic traumatic encephalopathy (CTE). The pipeline integrates multiple machine learning approaches including attention-based multiple instance learning (aMIL), semi-supervised learning, and novel clustering techniques to identify vascular injury patterns in young contact sports athletes.

## Research Background

This work analyzes a large collection of high-resolution digital whole slide images from an autopsy series of contact sports athletes, including adolescents and young adults. Our findings demonstrate that perivascular hemosiderin-laden macrophages are an early correlate of symptomatology following RHI, providing evidence for vascular injury as a biomarker preceding traditional CTE pathology.

## Key Components

### Core Pipeline Files
- `main.py` / `main2.py` - Primary analysis pipelines
- `annotation_expander.py` - Interactive semi-supervised learning tool for expanding labeled datasets
- `utils.py` - Utility functions for image processing and feature extraction
- `mlp_utils.py` - Multi-layer perceptron utilities for tile classification

### Feature Extraction & Analysis
- `extract_kan.py` - Feature extraction using KAN (Kolmogorov-Arnold Networks)
- `extract_trunks.py` - Feature extraction from image patches
- `centroid_kan.py` - Centroid-based analysis methods

### Clustering & Embedding
- **SPHERE** (Selective centroid Pooling with HDBSCAN and Embedding using Representative Extraction) - Novel deep learning modality for CTE prediction

## Interactive Notebooks

**Note**: Some notebooks are large and may not display directly on GitHub. Use these links to view them:

- [Example AnnotationExpander Notebook](https://nbviewer.org/github/Center-for-Computational-Neuropathology/Young-RHI-pipeline/blob/main/Example_AnnotationExpander.ipynb)
- [HemeMLP2 Notebook](https://nbviewer.org/github/Center-for-Computational-Neuropathology/Young-RHI-pipeline/blob/main/HemeMLP2.ipynb)
- [Vessels Extract Notebook](https://nbviewer.org/github/Center-for-Computational-Neuropathology/Young-RHI-pipeline/blob/main/VesselsExtractUMAPwHDBSCAN-KAN.ipynb)

## Features

- **Attention-based Multiple Instance Learning (aMIL)** - Cluster constrained model for WSI analysis
- **Semi-supervised Learning** - Interactive annotation expansion using UMAP and HDBSCAN clustering
- **Guided Attention Approximation** - High saliency region identification for improved CTE prediction
- **AI-guided Labeling** - Automated dataset generation for hemosiderin-laden macrophages and blood vessels
- **SPHERE Algorithm** - Novel approach for CTE prediction using hemosiderin burden analysis
- **Multi-scale Analysis** - Processing of 30+ million tiles across entire cohort

## Requirements

### Dependencies
```
python>=3.8
torch
torchvision
numpy
pandas
scikit-learn
umap-learn
hdbscan
openslide-python
skimage
matplotlib
seaborn
joblib
PIL
```

### Specialized Libraries
- `vision_transformer` (DINO features)
- `parametric_umap`

### System Requirements
- GPU recommended for deep learning components
- Minimum 16GB RAM for large WSI processing
- Substantial storage for histopathology data

## Installation

```bash
git clone https://github.com/username/Young_RHI_pipeline.git
cd Young_RHI_pipeline
pip install -r requirements.txt
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions about the methodology or collaboration opportunities, please open an issue or contact the research team.

## Disclaimer

This software is for research purposes only and not intended for clinical diagnosis. All analyses should follow appropriate IRB protocols and data use agreements.
