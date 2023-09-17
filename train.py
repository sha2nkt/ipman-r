from utils import TrainOptions
from train import Trainer
from loguru import logger
import argparse
from config import run_grid_search_experiments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--opts', default=[], nargs='*', help='additional options to update config')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
    parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
    parser.add_argument('--bid', type=int, default=30, help='amount of bid for cluster')
    parser.add_argument('--memory', type=int, default=20000, help='memory amount for cluster')
    parser.add_argument('--gpu_min_mem', type=int, default=11000, help='minimum amount of GPU memory')
    parser.add_argument('--gpu_arch', default=['tesla', 'quadro', 'rtx'],
                        nargs='*', help='additional options to update config')
    parser.add_argument('--num_cpus', type=int, default=8, help='num cpus for cluster')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        bid=args.bid,
        use_cluster=args.cluster,
        memory=args.memory,
        script='train.py',
        cmd_opts=args.opts,
        gpu_min_mem=args.gpu_min_mem,
        gpu_arch=args.gpu_arch,
    )
    trainer = Trainer(hparams)
    trainer.train()
