import os
os.environ["CDF_LIB"] = "/is/cluster/scratch/stripathi/data/cdf37_1-dist/src/lib"

import cv2
import json
import glob
import h5py
import torch
import trimesh
import constants
import numpy as np
import pickle as pkl
from tqdm import tqdm
from spacepy import pycdf
# from .read_openpose import read_openpose
import sys
sys.path.append('../../../')
from utils.geometry import batch_rodrigues, batch_rot2aa, ea2rm
from vis_utils.world_vis import overlay_mesh, vis_smpl_with_ground

joints_to_use = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 37])
joints_to_use = np.arange(0,156).reshape((-1,3))[joints_to_use].reshape(-1)

# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_train_extract(dataset_path, openpose_path, out_path, extract_img=False, vis_path=None, visualize=False):
    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, Ss_world_, openposes_  = [], [], [], [], [], [], []
    poses_, shapes_, transls_ = [], [], []
    poses_world_, transls_world_, cams_r_, cams_t_, cams_k_ = [], [], [], [], []
    in_bos_label_, contact_label_ = [], []
    # users in validation set
    user_list = [1, 5, 6, 7, 8]

    # go over each user
    for user_i in tqdm(user_list):
        user_name = 'S%d' % user_i
        print(user_name)
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, 'annotations_temp', user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose in world coordinates
        pose_world_path = os.path.join(dataset_path, 'annotations_temp', user_name, 'MyPoseFeatures', 'D3_Positions')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, 'annotations_temp', user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, 'annotations_temp', user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        # vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # path with videos
        imgs_path = os.path.join(dataset_path, 'Images')
        # path with mosh data
        mosh_path = os.path.join(dataset_path, 'mosh', user_name)

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for i, seq_i in enumerate(tqdm(seq_list)):
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            seq_world_i = os.path.join(pose_world_path, action+'.cdf')
            seq_world_name = seq_world_i.split('/')[-1]
            action = action.replace(' ', '_')

            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]
            poses_world_3d = pycdf.CDF(seq_world_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file, 'r')

            mosh_file = os.path.join(mosh_path, f'{action.replace("_", " ")}_poses.pkl')

            try:
                thetas, transls, thetas_world, transls_world, betas = read_mosh_data(mosh_file, cam=camera, user=user_name)
            except FileNotFoundError:
                print(f'{mosh_file} not found!!!')
                continue

            # camera info
            camR, camT, camK, cam_id = read_cam(cam=camera, user=user_name)

            # get stability labels: 1: stable, 0: unstable but in contact, -1: unstable and not in contact
            in_bos_labels, contact_labels = vis_smpl_with_ground(thetas_world, transls_world, betas, seq_world_name, vis_path,
                                 sub_sample=1,
                                 ground_offset=constants.GROUND_OFFSETS['h36m'],
                                 smpl_batch_size=9000,
                                 visualize=False)
            in_bos_labels = in_bos_labels.detach().cpu().numpy()
            contact_labels = contact_labels.detach().cpu().numpy()

            # visualize world smpl on ground plane
            if visualize:
                if user_name == 'S1' and cam_id == 1:
                    vis_smpl_with_ground(thetas_world, transls_world, betas, seq_world_name, vis_path)

            # video file
            # if extract_img:
            #     vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
            #     imgs_path = os.path.join(dataset_path, 'images')
            #     vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in tqdm(range(poses_3d.shape[0])):
                # read video frame
                # if extract_img:
                #     success, image = vidcap.read()
                #     if not success:
                #         break

                # check if you can keep this frame
                # if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                # image name
                # imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                imgname = os.path.join(dataset_path, 'Images', user_name, f'{action}.{camera}',
                                       f'{frame_i + 1:06d}.jpg')
                if user_name == 'S1' and 'TakingPhoto' in action:
                    temp_action = action.replace('TakingPhoto', 'Photo')
                    imgname = os.path.join(dataset_path, 'Images', user_name, f'{temp_action}.{camera}',
                                           f'{frame_i + 1:06d}.jpg')

                if user_name == 'S1' and 'WalkingDog' in action:
                    temp_action = action.replace('WalkingDog', 'WalkDog')
                    imgname = os.path.join(dataset_path, 'Images', user_name, f'{temp_action}.{camera}',
                                           f'{frame_i + 1:06d}.jpg')


                if not os.path.isfile(imgname):
                    print(imgname)
                    raise FileNotFoundError
                # save image
                if extract_img:
                    img_out = os.path.join(imgs_path, imgname)
                    cv2.imwrite(img_out, image)

                # read GT bounding box
                mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                ys, xs = np.where(mask==1)
                bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

                # read GT 3D pose
                partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                part17 = partall[h36m_idx]
                part = np.zeros([24,3])
                part[global_idx, :2] = part17
                part[global_idx, 2] = 1

                # read GT 3D pose
                Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                S17 = Sall[h36m_idx]
                S17 -= S17[0] # root-centered
                S24 = np.zeros([24,4])
                S24[global_idx, :3] = S17
                S24[global_idx, 3] = 1

                # read GT 3D pose in world coordinates
                Sall_world = np.reshape(poses_world_3d[frame_i, :], [-1, 3]) / 1000.
                S17_world = Sall_world[h36m_idx]
                # S17 -= S17[0]  # root-centered
                S24_world = np.zeros([24, 4])
                S24_world[global_idx, :3] = S17_world
                S24_world[global_idx, 3] = 1
                # if frame_i % 100 == 0:
                #     plot_3d_andCam(S24_world, np.array([[0,0,0]]))
                # TODO: run openpose for h36m
                # read openpose detections
                # json_file = os.path.join(openpose_path, 'coco',
                #     imgname.replace('.jpg', '_keypoints.json'))
                # openpose = read_openpose(json_file, part, 'h36m')
                openpose = np.zeros([25, 3])

                # read mosh data
                beta = betas[frame_i]
                theta = thetas[frame_i]
                transl = transls[frame_i]
                theta_world = thetas_world[frame_i]
                transl_world = transls_world[frame_i]
                cam_r = camR
                cam_t = camT
                cam_k = camK
                in_bos_label = in_bos_labels[frame_i]
                contact_label = contact_labels[frame_i]

                # store data
                imgnames_.append(imgname)
                centers_.append(center)
                scales_.append(scale)
                parts_.append(part)
                Ss_.append(S24)
                Ss_world_.append(S24_world)
                openposes_.append(openpose)
                poses_.append(theta)
                transls_.append(transl)
                poses_world_.append(theta_world)
                transls_world_.append(transl_world)
                shapes_.append(beta)
                cams_r_.append(cam_r)
                cams_t_.append(cam_t)
                cams_k_.append(cam_k)
                in_bos_label_.append(in_bos_label)
                contact_label_.append(contact_label)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'h36m_train_world.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_,
                       S_world=Ss_world_,
                       pose=poses_,
                       transl=transls_,
                       shape=shapes_,
                       openpose=openposes_,
                       pose_world=poses_world_,
                       transl_world=transls_world_,
                       cam_r=cams_r_,
                       cam_t=cams_t_,
                       cam_k=cams_k_,
                       in_bos_label=in_bos_label_,
                       contact_label=contact_label_,
             )
    print('Saved to ', out_file)

def load_camera_params( hf, path ):
    """Load h36m camera parameters
    Args
    hf: hdf5 open file with h36m cameras data
    path: path or key inside hf to the camera we are interested in
    Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
    """

    R = hf[ path.format('R') ][:]
    R = R.T

    T = hf[ path.format('T') ][:]
    f = hf[ path.format('f') ][:]
    c = hf[ path.format('c') ][:]
    k = hf[ path.format('k') ][:]
    p = hf[ path.format('p') ][:]

    name = hf[ path.format('Name') ][:]
    name = "".join( [chr(item) for item in name] )

    return R, T, f, c, k, p, name


def load_cameras(bpath='/ps/scratch/ps_shared/stripathi/frmKocabas/human36m/cameras.h5', subjects=[1,5,6,7,8,9,11]):
    """Loads the cameras of h36m
    Args
    bpath: path to hdf5 file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
    Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
    """
    rcams = {}

    with h5py.File(bpath,'r') as hf:
        for s in subjects:
            for c in range(4): # There are 4 cameras in human3.6m
                rcams[c+1] = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )
    return rcams

def load_cameras_json(bpath='/ps/scratch/ps_shared/stripathi/frmKocabas/human36m/h36m-camera-parameters.json'):
    """
    Loads the cameras from convenient json file at https://github.com/karfly/human36m-camera-parameters/blob/master/camera-parameters.json
    """
    with open(bpath) as f:
        data = json.load(f)
    return data

def read_pkl(f):
    return pkl.load(open(f, 'rb'), encoding='latin1')

def read_cam(cam=None, user=None):
    camera_ids = {
        '54138969': 1,
        '55011271': 2,
        '58860488': 3,
        '60457274': 4,
    }
    rcams_new = load_cameras_json()
    R = rcams_new['extrinsics'][user][cam]['R']
    T = rcams_new['extrinsics'][user][cam]['t']
    K = rcams_new['intrinsics'][cam]['calibration_matrix']
    cam_id = camera_ids[cam]
    # rcams = load_cameras()
    # cam_full = rcams[camera_ids[cam]] # order: R.T, T, f, c, k, p, name
    return R, T, K, cam_id

def read_mosh_data(f, frame_skip=4, cam=None, user=None):
    m2mm = 1000
    data = read_pkl(f)
    poses = data['pose_est_fullposes'][0::frame_skip,joints_to_use]
    poses_world = poses.copy()
    shape = data['shape_est_betas'][:10]
    shape = np.tile(shape, [poses.shape[0], 1])
    transls_world = data['pose_est_trans'][0::frame_skip]

    camera_ids = {
        '54138969': 1,
        '55011271': 2,
        '58860488': 3,
        '60457274': 4,
    }
    rcams_new = load_cameras_json()
    camR = rcams_new['extrinsics'][user][cam]['R']
    camT = rcams_new['extrinsics'][user][cam]['t']
    camPos = -np.matmul(np.array(camR).T, np.array(camT)) / m2mm  # C = -R.T x t

    # rcams = load_cameras()
    # camR = rcams[camera_ids[cam]][0]

    poses = torch.FloatTensor(poses)
    camR = torch.FloatTensor(camR).unsqueeze(0)

    poses[:,:3] = rectify_pose(camR, poses[:,:3])
    transl = transls_world - camPos.T
    return poses.numpy(), transl, poses_world, transls_world, shape


def rectify_pose(camera_r, body_aa):
    body_r = batch_rodrigues(body_aa).reshape(-1,3,3)
    final_r = camera_r @ body_r
    body_aa = batch_rot2aa(final_r)
    return body_aa