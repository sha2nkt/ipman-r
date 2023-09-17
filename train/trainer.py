import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
import ipdb

from datasets import MixedDataset
from models import hmr, SMPL
from smplify import SMPLify
from utils.geometry import batch_rot2aa, batch_rodrigues, perspective_projection, estimate_translation, \
    batch_rectify_pose, ea2rm
# from utils.renderer import Renderer
from utils.base_trainer import BaseTrainer

# for eval
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error, compute_accel, compute_error_accel
from utils.renderer import Renderer
# from utils.part_utils import PartRenderer
from stability.ground_losses import StabilityLossCoP, GroundLoss
from vis_utils.world_vis import vis_vert_with_ground

import config
import constants
from .fits_dict import FitsDict


class Trainer(BaseTrainer):

    def init_fn(self):
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)

        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH

        # Intuitive physics losses
        self.criterion_stability = StabilityLossCoP(
            faces = self.smpl.faces,
            cop_w = self.options.cop_w,
            cop_k = self.options.cop_k,
            contact_thresh = self.options.contact_thresh,
            model_type='smpl'
        )

        self.criterion_ground = GroundLoss(
            faces=self.smpl.faces,
            out_alpha1=self.options.out_alpha1,
            out_alpha2=self.options.out_alpha2,
            in_alpha1=self.options.in_alpha1,
            in_alpha2=self.options.in_alpha2,
            device=self.device,
            model_type='smpl'
        )

        # Initialize SMPLify fitting module
        self.smplify = SMPLify(step_size=1e-4, batch_size=self.options.batch_size,
                               num_iters=self.options.num_smplify_iters, focal_length=self.focal_length)
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Load dictionary of fits
        # self.fits_dict = FitsDict(self.options, self.train_ds)

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

    def finalize(self):
        self.fits_dict.save()

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def ip_losses(self, pred_rotmat, pred_betas, gt_cam_r, gt_joints_world, ground_offset, valid):
        """Intuitive physics losses"""

        ## bring predictions to world coordinates:
        # 1. apply GT cam on pred_pose
        # 2. compute tranlsation to align pred_pelvis to gt_pelvis
        # 3. rotate the vertices to align with y-axis (instead of z-axis in smpl)
        batch_size = pred_rotmat.shape[0]
        pred_rotmat_global_orient_world = torch.bmm(gt_cam_r.permute(0, 2, 1), pred_rotmat[:, 0])

        pred_output_world = self.smpl(betas=pred_betas,
                                    body_pose=pred_rotmat[:, 1:],
                                    global_orient=pred_rotmat_global_orient_world.unsqueeze(1),
                                      pose2rot=False)
        pred_vertices_world = pred_output_world.vertices
        pred_joints_world = pred_output_world.joints[:, 25:, :]

        # compute pred transl world by equating it to the gt world pelvis
        gt_joints_world = gt_joints_world[:, :, :-1] # removing the confidence values as always 1
        gt_world_pelvis = (gt_joints_world[:, 2, :] + gt_joints_world[:, 3, :]) / 2
        pred_pelvis_world = (pred_joints_world[:, 2, :] + pred_joints_world[:, 3, :]) / 2
        pred_transl_world = gt_world_pelvis - pred_pelvis_world

        pred_vertices_world += pred_transl_world[:, None, :]
        # rotate the vertices to align with the pyrender coordinates (yaxis-up), not aligned with y axis
        R1 = ea2rm(torch.tensor([[np.radians(270)]]), torch.tensor([[np.radians(0)]]),
                   torch.tensor([[np.radians(0)]])).float().to(self.device).expand(pred_vertices_world.shape[0], 3, 3)
        pred_vertices_world = torch.einsum('bki,bji->bjk', [R1, pred_vertices_world])
        ground_offset = torch.hstack([torch.zeros_like(ground_offset), ground_offset, torch.zeros_like(ground_offset)])
        pred_vertices_world += -ground_offset[:, None, :]

        # # rotate the vertices to align with the pyrender coordinates (yaxis-up), not aligned with y axis
        # gt_vertices_world = torch.einsum('bki,bji->bjk', [R1, gt_vertices_world])
        # gt_vertices_world += -ground_offset[:, None, :]
        #
        # import trimesh
        # mesh = trimesh.Trimesh(pred_vertices_world[0].detach().squeeze().cpu().numpy(), self.smpl.faces, process=False,
        #                        maintain_order=True)
        # mesh.export('pred_mesh_in_training.obj')
        # mesh = trimesh.Trimesh(gt_vertices_world[0].detach().squeeze().cpu().numpy(), self.smpl.faces, process=False,
        #                        maintain_order=True)
        # mesh.export('gt_mesh_in_training.obj')
        # import ipdb; ipdb.set_trace()


        # stability loss
        valid = valid.to(torch.bool)
        stability_loss = self.criterion_stability(pred_vertices_world)
        stability_loss = stability_loss[valid].mean()

        # ground loss
        out_pull_loss, in_push_loss  = self.criterion_ground(pred_vertices_world)
        out_pull_loss = out_pull_loss[valid].mean()
        in_push_loss = in_push_loss[valid].mean()
        return stability_loss, out_pull_loss, in_push_loss, pred_vertices_world


    def train_step(self, input_batch):
        self.model.train()

        # Get data from the batch
        images = input_batch['img']  # input image
        gt_keypoints_2d = input_batch['keypoints']  # 2D keypoints
        gt_pose = input_batch['pose']  # SMPL pose parameters
        gt_betas = input_batch['betas']  # SMPL beta parameters
        gt_joints = input_batch['pose_3d']  # 3D pose
        gt_cam_r = input_batch['cam_r'].to(torch.float32)
        gt_joints_world = input_batch['pose_3d_world'].to(torch.float32)
        gt_in_bos_label = input_batch['in_bos_label']
        gt_contact_label = input_batch['contact_label']
        has_smpl = input_batch['has_smpl'].byte()  # flag that indicates whether SMPL parameters are valid
        has_smpl_world = input_batch['has_smpl_world'].byte()  # flag that indicates whether SMPL world parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte()  # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped']  # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle']  # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name']  # name of the dataset the image comes from
        indices = input_batch['sample_index']  # index of example inside its dataset
        batch_size = images.shape[0]

        h36m_mask = torch.BoolTensor([1 if k == 'h36m' else 0 for k in dataset_name]).to(self.device)
        h36m_ground_offset = torch.FloatTensor([constants.GROUND_OFFSETS['h36m']]).to(self.device).to(torch.float32).expand(batch_size)
        ground_offset = input_batch['ground_offset'].to(torch.float32)
        ground_offset = ground_offset.masked_scatter_(h36m_mask, h36m_ground_offset)
        ground_offset = ground_offset[:, None]
        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        # gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        # gt_model_joints = gt_out.joints
        # gt_vertices = gt_out.vertices

        # # Get GT vertices and model joints in world coordinates
        # # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        # gt_out_world = self.smpl(betas=gt_betas,
        #                    body_pose=gt_pose_world[:, 3:],
        #                    global_orient=gt_pose_world[:, :3],
        #                    transl=gt_transl_world)
        # gt_vertices_world = gt_out_world.vertices

        # Get current best fits from the dictionary
        # opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        # opt_pose = opt_pose.to(self.device)
        # opt_betas = opt_betas.to(self.device)
        # opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:, 3:], global_orient=opt_pose[:, :3])
        # opt_vertices = opt_output.vertices
        # opt_joints = opt_output.joints

        # Get gt fits
        gt_pose = gt_pose.to(self.device)
        gt_betas = gt_betas.to(self.device)
        gt_output = self.smpl(betas=gt_betas,
                              body_pose=gt_pose[:, 3:],
                              global_orient=gt_pose[:, :3])
        gt_vertices = gt_output.vertices
        gt_model_joints = gt_output.joints

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # # Estimate camera translation given the model joints and 2D keypoints
        # # by minimizing a weighted least squares loss
        # gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length,
        #                                 img_size=self.options.img_res)


        gt_cam_t = estimate_translation(gt_model_joints,
                                        gt_keypoints_2d_orig,
                                        focal_length=self.focal_length,
                                        img_size=self.options.img_res,
                                        use_all_joints=True if '3dpw' in dataset_name else False)

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)
        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:, 1],
                                  pred_camera[:, 2],
                                  2 * self.focal_length / (self.options.img_res * pred_camera[:, 0] + 1e-9)], dim=-1)

        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(
                                                       batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        valid_fit = has_smpl
        # Add the examples with GT parameters to the list of valid fits
        valid_fit = valid_fit | has_smpl

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, valid_fit)

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, gt_vertices, valid_fit)

        # Intuitive physics losses
        if self.options.stability_loss_weight != 0 or \
            self.options.inside_push_loss_weight !=0 or\
            self.options.outside_pull_loss_weight !=0:
            # convert in_bos_label and contact label to one-hot (all unstable {0, -1} go to 0}
            gt_in_bos_label = (gt_in_bos_label == 1).byte()
            gt_contact_label = (gt_contact_label == 1).byte()
            valid_fit_stability = has_smpl & gt_in_bos_label & gt_contact_label
            loss_stability, loss_out_pull, loss_in_push, pred_vertices_world = self.ip_losses(pred_rotmat, pred_betas, gt_cam_r, gt_joints_world, ground_offset, valid_fit_stability)
        else:
            loss_stability = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_out_pull = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_in_push = torch.FloatTensor(1).fill_(0.).to(self.device)
            pred_vertices_world = pred_vertices

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.shape_loss_weight * loss_shape + \
               self.options.keypoint_loss_weight * loss_keypoints + \
               self.options.keypoint_loss_weight * loss_keypoints_3d + \
               self.options.pose_loss_weight * loss_regr_pose + \
               self.options.beta_loss_weight * loss_regr_betas + \
               self.options.stability_loss_weight * loss_stability + \
               self.options.inside_push_loss_weight * loss_in_push + \
               self.options.outside_pull_loss_weight * loss_out_pull + \
               ((torch.exp(-pred_camera[:, 0] * 10)) ** 2).mean()
        loss *= 60

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'pred_vertices_world': pred_vertices_world.detach(),
                  'opt_vertices': gt_vertices,
                  'pred_cam_t': pred_cam_t.detach(),
                  'opt_cam_t': gt_cam_t}
        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': self.options.keypoint_loss_weight * loss_keypoints.detach().item(),
                  'loss_keypoints_3d': self.options.keypoint_loss_weight * loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': self.options.pose_loss_weight * loss_regr_pose.detach().item(),
                  'loss_regr_betas': self.options.beta_loss_weight * loss_regr_betas.detach().item(),
                  'loss_shape': self.options.shape_loss_weight * loss_shape.detach().item(),
                  'loss_stability': self.options.stability_loss_weight * loss_stability.detach().item(),
                  'loss_in_push': self.options.inside_push_loss_weight * loss_in_push.detach().item(),
                  'loss_out_pull': self.options.outside_pull_loss_weight * loss_out_pull.detach().item()}
        return output, losses

    def train_summaries(self, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
        images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images)
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)


    def eval_step(self, epoch, dataset_name, dataset, result_file,
                       batch_size=32, img_res=224,
                       num_workers=32, shuffle=False, log_freq=None, step_count=-1):
        """Run evaluation on the datasets and metrics we report in the paper. """
        self.model.eval()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
            shuffle = False
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

        # Stability metrics
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
        tp = np.zeros((2, 1))
        fp = np.zeros((2, 1))
        fn = np.zeros((2, 1))
        parts_tp = np.zeros((7, 1))
        parts_fp = np.zeros((7, 1))
        parts_fn = np.zeros((7, 1))
        # Pixel count accumulators
        pixel_count = 0
        parts_pixel_count = 0

        # Store SMPL parameters
        smpl_pose = np.zeros((len(dataset), 72))
        smpl_betas = np.zeros((len(dataset), 10))
        smpl_camera = np.zeros((len(dataset), 3))
        pred_joints = np.zeros((len(dataset), 17, 3))

        eval_pose = False
        eval_masks = False
        eval_parts = False
        # Choose appropriate evaluation for each dataset
        if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' \
                or dataset_name == 'h36m-test-s1' or dataset_name == '3dpw' \
                or dataset_name == 'mpi-inf-3dhp' \
                or dataset_name == 'rich-val-onlycam0' or dataset_name == 'rich-test-onlycam0' \
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
            gt_pose = batch['pose'].to(device)
            gt_betas = batch['betas'].to(device)
            gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
            images = batch['img'].to(device)
            gender = batch['gender'].to(device)
            if 'rich' in dataset_name:
                ground_offset = batch['ground_offset'][:, None].to(device)
            elif 'h36m' in dataset_name:
                ground_offset = constants.GROUND_OFFSETS['h36m']
            else: #for 3dpw
                ground_offset = 0.0

            curr_batch_size = images.shape[0]

            # visualize mesh in world coordinates
            if len(torch.unique(batch['has_smpl_world'])) == 1 and torch.unique(batch['has_smpl_world'])[0] == 1:
                gt_pose_world = batch['pose_world'].to(device)
                gt_transl_world = batch['transl_world'].to(device)
                gt_body_world = smpl_neutral(betas=gt_betas,
                                             body_pose=gt_pose_world[:, 3:],
                                             global_orient=gt_pose_world[:, :3],
                                             transl=gt_transl_world)

                gt_joints_world = gt_body_world.joints[:, 25:, :]
                # visualize the gt vertices in world coords
                gt_in_bos_label, gt_contact_metric, gt_contact_mask = vis_vert_with_ground(gt_body_world.vertices,
                                                                              gt_transl_world[:, None, :]*0.0,
                                                                              seq_name='test-gt_vert',
                                                                              start_idx=step * curr_batch_size,
                                                                              sub_sample=1,
                                                                              smpl_batch_size=curr_batch_size,
                                                                              ground_offset=ground_offset,
                                                                              visualize=False)
                if gt_in_bos_label.dim() == 0:
                    gt_in_bos_label = gt_in_bos_label.view(-1)
                    gt_contact_metric = gt_contact_metric.view(-1)
                    gt_contact_mask = gt_contact_mask.view(1, -1)

                gt_bos_accumulator.append(gt_in_bos_label)
                gt_contact_accumulator.append(gt_contact_metric)
                gt_contact_mask_accumulator.append(gt_contact_mask)

                has_smpl_world = True
            else:
                gt_joints_world = batch['pose_3d_world'][:, :, :-1].to(device)
                has_smpl_world = False

            # Get ground truth camera
            gt_cam_r = batch['cam_r'].to(device).to(torch.float32)

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = self.model(images)
                pred_pose = batch_rot2aa(pred_rotmat.view(-1, 3, 3)).view(curr_batch_size, -1)
                pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                           global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
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
                pred_in_bos_label, pred_contact_metric, pred_contact_mask = vis_vert_with_ground(pred_vertices_world,
                                                                              pred_transl_world[:, None, :],
                                                                              seq_name='test-pred',
                                                                              start_idx=step * curr_batch_size,
                                                                              sub_sample=1,
                                                                              smpl_batch_size=curr_batch_size,
                                                                              ground_offset=ground_offset,
                                                                              visualize=False)

                if pred_in_bos_label.dim() == 0:
                    pred_in_bos_label = pred_in_bos_label.view(-1)
                    pred_contact_metric = pred_contact_metric.view(-1)
                    pred_contact_mask = pred_contact_mask.view(1, -1)

                pred_bos_accumulator.append(pred_in_bos_label)
                pred_contact_accumulator.append(pred_contact_metric)
                pred_contact_mask_accumulator.append(pred_contact_mask)

            if save_results:
                rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
                rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
                pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
                smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
                smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :] = pred_betas.cpu().numpy()
                smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :] = pred_camera.cpu().numpy()

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
                    gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                            betas=gt_betas).vertices
                    gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                     betas=gt_betas).vertices
                    gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                    gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                    gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
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
                    pred_joints[step * batch_size:step * batch_size + curr_batch_size, :,
                    :] = pred_keypoints_3d.cpu().numpy()
                pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
                pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
                pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
                pred_vertices = pred_vertices - pred_pelvis

                # Absolute error (MPJPE)
                error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

                # Reconstuction_error
                r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                               reduction=None)
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
                # mask, parts = renderer(pred_vertices, pred_camera)

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
                        fp[c] += (~cgt & cpred).sum()
                        fn[c] += (cgt & ~cpred).sum()
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
                        parts_fp[c] += (~cgt & cpred).sum()
                        parts_fn[c] += (cgt & ~cpred).sum()
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
            #         print()
            #     if eval_masks:
            #         print('Accuracy: ', accuracy / pixel_count)
            #         print('F1: ', f1.mean())
            #         print()
            #     if eval_parts:
            #         print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
            #         print('Parts F1 (BG): ', parts_f1[[0, 1, 2, 3, 4, 5, 6]].mean())
            #         print()

        # find accuracy of bos metric
        pred_bos_accumulator = torch.cat(pred_bos_accumulator, dim=0).to(torch.float32)
        pred_contact_accumulator = torch.cat(pred_contact_accumulator, dim=0).to(torch.float32)
        pred_contact_mask_accumulator = torch.cat(pred_contact_mask_accumulator, dim=0).to(torch.float32)
        print('*** Stability Metrics ***')
        print()
        if has_smpl_world:
            gt_bos_accumulator = torch.cat(gt_bos_accumulator, dim=0)
            pred_bos_accuracy = (gt_bos_accumulator == pred_bos_accumulator).sum().item() / \
                                gt_bos_accumulator.shape[0]
            gt_contact_accumulator = torch.cat(gt_contact_accumulator, dim=0)
            gt_contact_mask_accumulator = torch.cat(gt_contact_mask_accumulator, dim=0)
            pred_contact_accuracy = (gt_contact_accumulator == pred_contact_accumulator).sum().item() / \
                                    gt_contact_accumulator.shape[0]
            dense_contact_accuracy =  (gt_contact_mask_accumulator == pred_contact_mask_accumulator).sum().item() / \
                                      gt_contact_mask_accumulator.numel()
            print(f'{dataset_name} BOS Accuracy: ', pred_bos_accuracy)
            print(f'{dataset_name} Contact Accuracy: ', pred_contact_accuracy)
            print(f'{dataset_name} Contact Mask Accuracy: ', dense_contact_accuracy)
            self.summary_writer.add_scalar(f'{dataset_name} BOS Accuracy', pred_bos_accuracy, epoch)
            self.summary_writer.add_scalar(f'{dataset_name} Contact Accuracy', pred_contact_accuracy, epoch)
            self.summary_writer.add_scalar(f'{dataset_name} Contact Mask Accuracy', dense_contact_accuracy, epoch)
        else:
            bos_mean = torch.mean(pred_bos_accumulator)
            contact_mean = torch.mean(pred_contact_accumulator)
            print(f'{dataset_name} BOS Mean: ', bos_mean.cpu().numpy())
            print(f'{dataset_name} Contact Mean: ', contact_mean.cpu().numpy())
            self.summary_writer.add_scalar(f'{dataset_name} BOS Mean', bos_mean.cpu().numpy(), epoch)
            self.summary_writer.add_scalar(f'{dataset_name} Contact Mean', contact_mean.cpu().numpy(), epoch)

        # Save reconstructions to a file for further processing
        if save_results:
            np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
        # Print final results during evaluation
        print('*** Final Results ***')
        print()
        if eval_pose:
            print(f'{dataset_name} MPJPE: ' + str(1000 * mpjpe.mean()))
            print(f'{dataset_name} Reconstruction Error: ' + str(1000 * recon_err.mean()))
            print(f'{dataset_name} V2V Error: ' + str(1000 * v2v_err.mean()))
            print(f'{dataset_name} Accel: ' + str(1000 * accel_.mean()))
            print(f'{dataset_name} Accel Error: ' + str(1000 * accel_err_.mean()))
            print()
            self.summary_writer.add_scalar(f'{dataset_name} MPJPE', 1000 * mpjpe.mean(), epoch)
            self.summary_writer.add_scalar(f'{dataset_name} Reconstruction Error', 1000 * recon_err.mean(), epoch)
            self.summary_writer.add_scalar(f'{dataset_name} V2V Error', 1000 * v2v_err.mean(), epoch)
            self.summary_writer.add_scalar(f'{dataset_name} Accel', 1000 * accel_.mean(), epoch)
            self.summary_writer.add_scalar(f'{dataset_name} Acceleration Error', 1000 * accel_err_.mean(), epoch)
        if eval_masks:
            print('Accuracy: ', accuracy / pixel_count)
            print('F1: ', f1.mean())
            print()
        if eval_parts:
            print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
            print('Parts F1 (BG): ', parts_f1[[0, 1, 2, 3, 4, 5, 6]].mean())



