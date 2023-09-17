import os
import sys
import stat
import shutil
import subprocess

from loguru import logger

GPUS = {
    'v100-v16': ('\"Tesla V100-PCIE-16GB\"', 'tesla', 16000),
    'v100-p32': ('\"Tesla V100-PCIE-32GB\"', 'tesla', 32000),
    'v100-s32': ('\"Tesla V100-SXM2-32GB\"', 'tesla', 32000),
    'v100-p16': ('\"Tesla P100-PCIE-16GB\"', 'tesla', 16000),
    'rtx2080ti': ('\"NVIDIA GeForce RTX 2080 Ti\"', 'rtx', 11000),
    'quadro6000': ('\"Quadro RTX 6000\"', 'quadro', 24000),
}

def get_gpus(min_mem=10000, arch=('tesla', 'quadro', 'rtx')):
    gpu_names = []
    for k, (gpu_name, gpu_arch, gpu_mem) in GPUS.items():
        if gpu_mem >= min_mem and gpu_arch in arch:
            gpu_names.append(gpu_name)

    assert len(gpu_names) > 0, 'Suitable GPU model could not be found'

    return gpu_names


def execute_task_on_cluster(
        script,
        exp_name,
        input_folder,
        cluster_bs,
        num_queue,
        condor_dir,
        cfg_file,
        exp_opts=None,
        bid_amount=5,
        num_workers=8,
        memory=64000,
        gpu_min_mem=10000,
        gpu_arch=('tesla', 'quadro', 'rtx'),
        num_gpus=1
):
    # copy config to a new experiment directory and source from there
    temp_config_dir = os.path.join(os.path.dirname(condor_dir), 'temp_configs')
    logdir = os.path.join(temp_config_dir, f'{exp_name}')
    os.makedirs(logdir, exist_ok=True)
    new_cfg_file = os.path.join(logdir, 'config.yaml')
    shutil.copy(src=cfg_file, dst=new_cfg_file)

    gpus = get_gpus(min_mem=gpu_min_mem, arch=gpu_arch)

    gpus = ' || '.join([f'CUDADeviceName=={x}' for x in gpus])

    os.makedirs(os.path.join(condor_dir, 'condorlog', exp_name), exist_ok=True)
    submission = f'executable = {condor_dir}/condorlog/{exp_name}/{exp_name}_run.sh\n' \
                 'arguments = $(Process) $(Cluster)\n' \
                 f'error = {condor_dir}/condorlog/{exp_name}/$(Cluster).$(Process).err\n' \
                 f'output = {condor_dir}/condorlog/{exp_name}/$(Cluster).$(Process).out\n' \
                 f'log = {condor_dir}/condorlog/{exp_name}/$(Cluster).$(Process).log\n' \
                 f'request_memory = {memory}\n' \
                 f'request_cpus={int(num_workers/2)}\n' \
                 f'request_gpus={num_gpus}\n' \
                 f'requirements={gpus}\n' \
                 f'+MaxRunningPrice = 500\n' \
                 f'queue {num_queue}'
                 # f'request_cpus={int(num_workers/2)}\n' \
                 # f'+RunningPriceExceededAction = \"kill\"\n' \
    print('<<< Condor Submission >>> ')
    print(submission)

    with open('submit.sub', 'w') as f:
        f.write(submission)

    # output_dir = os.path.join(output_dir, exp_name)
    logger.info(f'The logs for this experiments can be found under: {condor_dir}/condorlog/{exp_name}')
    bash = 'export PYTHONBUFFERED=1\n export PATH=$PATH\n ' \
           f'{sys.executable} ../ipman_regr/datasets/preprocess/RICH/{script} ' \
           f'--config {new_cfg_file} ' \
           f'--input_folder {input_folder} ' \
           f' --condor_dir {condor_dir} ' \
           f' --cluster_bs {cluster_bs} ' \
           f' --ds_start_idx $1'  ## This is the trick. Notice there is no --cluster here
    if exp_opts is not None:
        bash += ' --opts '
        for opt in exp_opts:
            bash += f'{opt} '
        bash += 'SYSTEM.CLUSTER_NODE $2.$1'
    else:
        bash += ' --opts SYSTEM.CLUSTER_NODE $2.$1'

    executable_path =f'{condor_dir}/condorlog/{exp_name}/{exp_name}_run.sh'

    with open(executable_path, 'w') as f:
        f.write(bash)

    os.chmod(executable_path, stat.S_IRWXU)

    cmd = ['condor_submit_bid', f'{bid_amount}', 'submit.sub']
    logger.info('Executing ' + ' '.join(cmd))
    subprocess.call(cmd)
