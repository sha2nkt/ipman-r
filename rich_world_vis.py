# take a human3.6 file and visualize skeletons in world coordinates

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets.preprocess.rich import rich_extract
import os
# for cluster rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

if __name__ == '__main__':
    # load argparse and take data_path as input
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/ps/scratch/ps_shared/stripathi/frmKocabas/human36m',
                        help='path to the h36m dataset')
    parser.add_argument('--split', type=str, default='val',
                        help='test or val split')
    parser.add_argument('--out_path', type=str, default='data/dataset_extras/',
                        help='path to the h36m dataset')
    parser.add_argument('--vis_path', type=str, default='data/debug_vis/',
                        help='path to the data where visualizations are stored')
    parser.add_argument('--visualize', default=False, action='store_true',
                        help='generate visualization?')
    args = parser.parse_args()

    rich_extract(dataset_path=args.data_path,
                       split=args.split,
                       out_path=args.out_path,
                       vis_path=args.vis_path,
                       visualize=args.visualize)

