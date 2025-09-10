#!/bin/bash
#BSUB -J "test"
#BSUB -P acc_tauomics
#BSUB -q gpu
#BSUB -R v100
#BSUB -gpu num=1 
#BSUB -R rusage[mem=128000]
#BSUB -n 1
#BSUB -W 144:00
#BSUB -oo /sc/arion/projects/tauomics/danielk/AnnotationExpander/minerva_out/test_out.txt
#BSUB -eo /sc/arion/projects/tauomics/danielk/AnnotationExpander/minerva_out/test_error.txt
ml cuda cudnn 
cd /sc/arion/projects/tauomics/danielk/
source activate GPUClean
python /sc/arion/projects/tauomics/danielk/AnnotationExpander/test.py \
--feat_dir /sc/arion/projects/tauomics/FeatureVectors/ParkmanRHI/dino