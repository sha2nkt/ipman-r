"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
# for cluster rendering
import argparse
import trimesh
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm
import pprint


import config
import constants
from models import hmr, SMPL
from datasets import BaseDatasetEval
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error, compute_accel, compute_error_accel
from utils.geometry import batch_rodrigues, batch_rot2aa, ea2rm, rot6d_to_rotmat, batch_rectify_pose

# from utils.part_utils import PartRenderer
from vis_utils.world_vis import vis_smpl_with_ground, vis_vert_with_ground

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'h36m-test-s1', 'h36m-train-small', 'lsp', '3dpw', 'mpi-inf-3dhp',
                                                             'rich-val-onlycam0', 'rich-val-onlycam1', 'rich-val-onlycam5', 'rich-test-last2seq-onlycam0',
                                                             'rich-test-onlycam0', 'rich-val', 'rich-test'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--vis_path', default=None, help='save the world frame visualizations here')
parser.add_argument('--visualize', default=False, action='store_true', help='generate visualization?')

def cam_to_world(vertices, translation, cam_r, cam_t):
    """
    Convert vertices from camera coordinates to world coordinates.
    """
    # apply predicted translation to the mesh (as tb = -tc)
    vertices = vertices + translation[:, None, :]

    cam_r = cam_r.to(torch.float32)
    cam_t = cam_t.to(torch.float32)

    # cam extrinsics
    mm2m = 1000
    R = cam_r
    t = -torch.bmm(R, cam_t) / mm2m  # t= -RC
    # t = cam_t.to(torch.float32) / mm2m

    # reverse camera to go from camera to world
    R_T = R.permute(0, 2, 1)
    t_w = -torch.bmm(R_T, t)

    # apply extrinsics
    vertices_world = torch.einsum('bij,bkj->bki', R_T, vertices)
    vertices_world = vertices_world + t_w.squeeze()[:, None, :]
    return vertices_world

def run_evaluation(model, dataset_name, dataset, result_file, vis_path,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50, visualize=False):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)

    # renderer = PartRenderer()

    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))
    v2v_err = np.zeros(len(dataset))

    # Acceleration metrics
    accel_ = np.zeros(len(dataset))
    accel_err_ = np.zeros(len(dataset))

    gt_bos_accumulator = []
    gt_contact_accumulator = []
    gt_contact_mask_accumulator = []
    pred_bos_accumulator = []
    pred_contact_accumulator = []
    pred_contact_mask_accumulator = []

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # classwise error Metrics
    classwise_mpjpe = {}

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2'  \
            or dataset_name == 'h36m-test-s1' or dataset_name == '3dpw' \
            or dataset_name == 'mpi-inf-3dhp' \
            or dataset_name == 'rich-val-onlycam0' or dataset_name == 'rich-test-onlycam0'\
            or dataset_name == 'rich-val' or dataset_name == 'rich-test':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        if len(torch.unique(batch['has_smpl'])) == 1 and torch.unique(batch['has_smpl'])[0] == 1:
            gt_pose = batch['pose'].to(device)
            gt_betas = batch['betas'].to(device)
            gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
            has_smpl = True

        images = batch['img'].to(device)
        imgnames = batch['imgname']
        gender = batch['gender'].to(device)
        curr_batch_size = images.shape[0]
        if 'rich' in dataset_name:
            ground_offset = batch['ground_offset'][:, None].to(device)
        elif 'h36m' in dataset_name:
            ground_offset = constants.GROUND_OFFSETS['h36m']
        else:
            ground_offset = 0
        # visualize mesh in world coordinates
        if len(torch.unique(batch['has_smpl_world'])) == 1 and torch.unique(batch['has_smpl_world'])[0] == 1:
            gt_pose_world = batch['pose_world'].to(device)
            gt_transl_world = batch['transl_world'].to(device)
            gt_body_world = smpl_neutral(betas=gt_betas,
                                         body_pose=gt_pose_world[:, 3:],
                                         global_orient=gt_pose_world[:, :3],
                                         transl=gt_transl_world)

            gt_joints_world = gt_body_world.joints[:, 25:, :]
            # # visualize the gt vertices in world coords
            # gt_in_bos_label, gt_contact_metric, gt_contact_mask = vis_vert_with_ground(gt_body_world.vertices,
            #                                                               gt_transl_world[:, None, :]*0.0, # since already included in smpl
            #                                                               seq_name='test-gt_vert',
            #                                                               vis_path=vis_path,
            #                                                               start_idx=step * curr_batch_size,
            #                                                               sub_sample=1,
            #                                                               smpl_batch_size=curr_batch_size,
            #                                                               imgnames = imgnames,
            #                                                               ground_offset=ground_offset,
            #                                                               visualize=False)
            # # gt_in_bos_label, gt_contact_metric = vis_smpl_with_ground(gt_pose_world, gt_transl_world, gt_betas,
            # #                                                           seq_name='test-gt_pose',
            # #                                                           vis_path=vis_path,
            # #                                                           start_idx= step * curr_batch_size,
            # #                                                           sub_sample=1,
            # #                                                           smpl_batch_size=curr_batch_size,
            # #                                                           ground_offset = ground_offset,
            # #                                                           visualize=visualize)
            # gt_bos_accumulator.append(gt_in_bos_label)
            # gt_contact_accumulator.append(gt_contact_metric)
            # gt_contact_mask_accumulator.append(gt_contact_mask)
            has_smpl_world = True
        else:
            gt_joints_world = batch['pose_3d_world'][:, :, :-1].to(device)
            has_smpl_world = False



        # Get ground truth camera
        gt_cam_r = batch['cam_r'].to(device).to(torch.float32)
        gt_cam_t = batch['cam_t'].to(device).to(torch.float32)
        gt_cam_k = batch['cam_k'].to(device).to(torch.float32)

        # # visualize mesh by going from camera to world coordinates
        # # apply reverse rotation to pose in camera coordinates
        # gt_pose_new = gt_pose.clone()
        # gt_pose_new[:,:3] = batch_rectify_pose(gt_cam_r.permute(0, 2, 1), gt_pose_new[:,:3])
        # vis_smpl_with_ground(gt_pose_new, gt_transl_world, gt_betas,
        #                      seq_name='test-gt-recovered',
        #                      vis_path=vis_path,
        #                      sub_sample=1,
        #                      ground_offset = ground_offset,
        #                      smpl_batch_size=curr_batch_size)

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_pose = batch_rot2aa(pred_rotmat.view(-1, 3, 3)).view(curr_batch_size, -1)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

            ## move to world coordinates - first correct the predicted pose using gt camera rotation, then correct the translation by pelvis alignment
            pred_pose_world = pred_pose.clone()
            pred_pose_world[:, :3] = batch_rectify_pose(gt_cam_r.permute(0, 2, 1), pred_pose_world[:, :3])
            pred_output_world = smpl_neutral(betas=pred_betas,
                                             body_pose=pred_pose_world[:, 3:],
                                             global_orient=pred_pose_world[:, :3])
            pred_vertices_world = pred_output_world.vertices
            pred_joints_world = pred_output_world.joints[:, 25:, :]
            # compute pred transl world by equating it to the gt world pelvis
            gt_world_pelvis = (gt_joints_world[:, 2, :] + gt_joints_world[:, 3, :]) / 2
            pred_pelvis_world = (pred_joints_world[:, 2, :] + pred_joints_world[:, 3, :]) / 2
            pred_transl_world = gt_world_pelvis - pred_pelvis_world
            pred_in_bos_label, pred_contact_metric, pred_contact_mask = vis_vert_with_ground(pred_vertices_world, pred_transl_world[:, None, :],
                                                                          seq_name='test-pred-offset',
                                                                          vis_path=vis_path,
                                                                          start_idx=step * curr_batch_size,
                                                                          sub_sample=1,
                                                                          imgnames = imgnames,
                                                                          smpl_batch_size=curr_batch_size,
                                                                          ground_offset=ground_offset,
                                                                          visualize=visualize)
            pred_bos_accumulator.append(pred_in_bos_label)
            pred_contact_accumulator.append(pred_contact_metric)
            pred_contact_mask_accumulator.append(pred_contact_mask)

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()

        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            if '3dpw' in dataset_name:
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            if 'rich' in dataset_name:
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
                gt_vertices = gt_vertices - gt_pelvis

                # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            pred_vertices = pred_vertices - pred_pelvis

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error
            for err, imgname in zip(error, imgnames):
                class_name = imgname.split('/')[-2].split('.')[0].split('_')[0]
                classwise_mpjpe.setdefault(class_name, []).append(err)

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # V2V error in RICH
            if 'rich' in dataset_name:
                v2v = torch.sqrt(((gt_vertices - pred_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                v2v_err[step * batch_size:step * batch_size + curr_batch_size] = v2v

            # Acceleration Error
            if pred_keypoints_3d.shape[0] >= 3: # can only calculate accel if there are at least 3 frames
                accel = np.mean(compute_accel(pred_keypoints_3d.cpu().numpy()))
                if not np.isnan(accel):
                    accel_[step * batch_size:step * batch_size + curr_batch_size] = accel
                accel_err = np.mean(compute_error_accel(joints_pred=pred_keypoints_3d.cpu().numpy(), joints_gt=gt_keypoints_3d.cpu().numpy()))
                if not np.isnan(accel_err):
                    accel_err_[step * batch_size:step * batch_size + curr_batch_size] = accel_err

        # If mask or part evaluation, render the mask and part images
        # if eval_masks or eval_parts:
        #     mask, parts = renderer(pred_vertices, pred_camera)

        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] +=  (~cgt & cpred).sum()
                    fn[c] +=  (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                    cgt = gt_parts == c
                    cpred = pred_parts == c
                    cpred[gt_parts == 255] = 0
                    parts_tp[c] += (cgt & cpred).sum()
                    parts_fp[c] +=  (~cgt & cpred).sum()
                    parts_fn[c] +=  (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

        # # Print intermediate results during evaluation
        # if step % log_freq == log_freq - 1:
        #     if eval_pose:
        #         print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
        #         print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
        #         print('V2V Error: ' + str(1000 * v2v_err[:step * batch_size].mean()))
        #         print()
        #     if eval_masks:
        #         print('Accuracy: ', accuracy / pixel_count)
        #         print('F1: ', f1.mean())
        #         print()
        #     if eval_parts:
        #         print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        #         print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        #         print()

    # find accuracy of bos metric
    pred_bos_accumulator = torch.cat(pred_bos_accumulator, dim=0).to(torch.float32)
    pred_contact_accumulator = torch.cat(pred_contact_accumulator, dim=0).to(torch.float32)
    pred_contact_mask_accumulator = torch.cat(pred_contact_mask_accumulator, dim=0).to(torch.float32)
    print('*** Stability Metrics ***')
    print()
    if has_smpl_world:
        gt_bos_accumulator = torch.cat(gt_bos_accumulator, dim=0)
        pred_bos_accuracy = (gt_bos_accumulator == pred_bos_accumulator).sum().item() / gt_bos_accumulator.shape[0]
        gt_contact_accumulator = torch.cat(gt_contact_accumulator, dim=0)
        gt_contact_mask_accumulator = torch.cat(gt_contact_mask_accumulator, dim=0)
        pred_contact_accuracy = (gt_contact_accumulator == pred_contact_accumulator).sum().item() / gt_contact_accumulator.shape[0]
        dense_contact_accuracy = (gt_contact_mask_accumulator == pred_contact_mask_accumulator).sum().item() / \
                                 gt_contact_mask_accumulator.numel()
        print('BOS Accuracy: ', pred_bos_accuracy)
        print('Contact Accuracy: ', pred_contact_accuracy)
        print(f'Contact Mask Accuracy: ', dense_contact_accuracy)
    else:
        bos_mean = torch.mean(pred_bos_accumulator)
        contact_mean = torch.mean(pred_contact_accumulator)
        print('BOS Mean: ', bos_mean.cpu().numpy())
        print('Contact Mean: ', contact_mean.cpu().numpy())

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('V2V Error: ' + str(1000 * v2v_err.mean()))
        print('---Classwise MPJPE---')
        for k, v in classwise_mpjpe.items():
            classwise_mpjpe[k] = str(1000 * np.array(v).mean())
        pprint.pprint(classwise_mpjpe)
        print()
    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()
    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        print()

if __name__ == '__main__':
    args = parser.parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset = BaseDatasetEval(None, args.dataset, is_train=False)
    # Run evaluation
    run_evaluation(model, args.dataset, dataset, args.result_file, args.vis_path,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq,
                   num_workers=args.num_workers,
                   visualize=args.visualize)
