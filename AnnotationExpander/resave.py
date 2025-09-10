from utils import * 
from os import listdir
from os.path import join 

import argparse 

parser = argparse.ArgumentParser(description='Resave as hashmaps')
parser.add_argument('--feat_dir', type=str, default=None)
args = parser.parse_args()

for slide in listdir(join(args.feat_dir, 'coords')): 
    slide = slide.replace('.pt', '')
    resaveFeatsAndCoords(slide, args.feat_dir)