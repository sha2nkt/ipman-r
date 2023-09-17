"""
python ../ipman_regr/datasets/preprocess//RICH/rich_smplx2smpl_conversion.py \
 --condor_dir ../ipman_regr/datasets/preprocess//RICH/condor \
 --input_folder /ps/scratch/ps_shared/stripathi/4yogi/RICH/val/2021-06-15_Multi_IOI_ID_03588_Yoga1/params \
 --config ../ipman_regr/datasets/preprocess//RICH/rich_config.yaml \
  --exp_name 2021-06-15_Multi_IOI_ID_03588_Yoga1 \
  --ds_start_idx 0 \
  --num_queue 19 \
  --cluster_bs 50 \
  --cluster
"""

import os
import sys
import subprocess
import argparse
import glob
from rich_config import run_grid_search_experiments
import shutil
import time
import numpy as np
from PIL import Image
import multiprocessing as mp




def main(args, downscale_factor=4):
    in_dir = args.input_folder
    # recursively get all pkl files in the directory
    types = ('*.png', '*.bmp')  # the tuple of file types
    in_images = []
    for files in types:
        in_images.extend(glob.glob(os.path.join(in_dir, '*/*/images_orig/'+files)))
    # select index for cluster usage
    print('Input Folder has {} objects'.format(len(in_images)))

    idxs = np.arange(len(in_images))
    cbs = len(in_images) if args.get('cluster_bs') is None else args.get('cluster_bs')
    sidx = args.ds_start_idx
    for idx in idxs[sidx * cbs: sidx * cbs + cbs]:
        in_img = in_images[idx]
        out_img_dir = os.path.dirname(in_img).replace('images_orig', 'images_orig_720p')
        os.makedirs(out_img_dir, exist_ok=True)
        out_img = os.path.join(out_img_dir, os.path.basename(in_img).replace('.bmp', '.png'))
        # bash = 'export PYTHONBUFFERED=1\n export PATH=$PATH\n' \
        #        f'{sys.executable} {script} --cfg {new_cfg_file} --cfg_id $1'
        img = Image.open(in_img)
        new_w = img.width // downscale_factor
        new_h = img.height // downscale_factor
        img = img.resize((new_w, new_h), Image.ANTIALIAS)
        print(f'Saving {out_img}')
        img.save(out_img)

def add_common_cmdline_args(parser):
    parser.add_argument('--opts', default=[], nargs='*', help='additional options to update config')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
    parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
    parser.add_argument('--bid', type=int, default=30, help='amount of bid for cluster')
    parser.add_argument('--memory', type=int, default=10000, help='memory amount for cluster')
    parser.add_argument('--gpu_min_mem', type=int, default=10000, help='minimum amount of GPU memory')
    parser.add_argument('--gpu_arch', default=['tesla', 'quadro', 'rtx'],
                        nargs='*', help='additional options to update config')
    parser.add_argument('--num_cpus', type=int, default=1, help='num cpus for cluster')
    return parser


if __name__ == '__main__':
    # load argparse and take data_path as input
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='/ps/scratch/ps_shared/stripathi/frmKocabas/human36m',
                        help='path to the h36m dataset')
    # config file added in common agrugments below
    parser.add_argument('-c', '--config',
                        required=True,
                        help='config file path')
    parser.add_argument('--exp_name',
                        default=None,
                        help='Custom name of the experiment')
    parser.add_argument('--ds_start_idx', type=int, default=0,
                        help='set index at which to start processing dataset')
    parser.add_argument('--cluster_bs', type=int, default=1,
                        help='number of dataset objects to process')
    parser.add_argument('--condor_dir',
                        default='./condorlog',
                        type=str,
                        help='The folder where the condor logs and configs are stored.')
    parser.add_argument('--num_queue',
                        default=1,
                        help='The number of jobs in db_file to run in cluster')
    parser = add_common_cmdline_args(parser)
    args = parser.parse_args()
    args = vars(args)
    args = run_grid_search_experiments(
        args,
        script='rich_downsample_images.py',
    )
    main(args)

