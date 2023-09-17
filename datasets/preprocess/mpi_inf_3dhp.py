import os
import sys
import cv2
import glob
import h5py
import json
import numpy as np
import scipy.io as sio
import pandas as pd
import scipy.misc
from .read_openpose import read_openpose

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts
    
def train_data(dataset_path, openpose_path, out_path, joints_idx, scaleFactor, extract_img=False, fits_3d=None, vis_path=None, visualize=None):

    joints17_idx = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]

    h, w = 2048, 2048
    imgnames_, scales_, centers_ = [], [], []
    parts_, Ss_, Ss_world_, openposes_ = [], [], [], []
    poses_, shapes_, transls_ = [], [], []
    poses_world_, transls_world_, cams_r_, cams_t_, cams_k_ = [], [], [], [], []
    in_bos_label_, contact_label_ = [], []


    # training data
    user_list = range(1,9)
    seq_list = range(1,3)
    vid_list = list(range(3)) + list(range(4,9))

    counter = 0

    # load already prepared npz for 3d fits
    npz_path = '/is/cluster/scratch/stripathi/pycharm_remote/ipman_regr/data/dataset_extras/mpi_inf_3dhp_train.npz'
    mpi_npz = np.load(npz_path)
    mpi_df = pd.DataFrame.from_dict({item: mpi_npz[item] for item in mpi_npz.files}, orient='index').T

    for user_i in user_list:
        for seq_i in seq_list:
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            calib_file = os.path.join(seq_path, 'camera.calibration')
            Ks, Rs, Ts = read_calibration(calib_file, vid_list)

            for j, vid_i in enumerate(vid_list):

                # image folder
                imgs_path = os.path.join(seq_path,
                                         'imageFrames',
                                         'video_' + str(vid_i))

                # extract frames from video file
                if extract_img:

                    # if doesn't exist
                    if not os.path.isdir(imgs_path):
                        os.makedirs(imgs_path)

                    # video file
                    vid_file = os.path.join(seq_path,
                                            'imageSequence',
                                            'video_' + str(vid_i) + '.avi')
                    vidcap = cv2.VideoCapture(vid_file)

                    # process video
                    frame = 0
                    while 1:
                        # extract all frames
                        success, image = vidcap.read()
                        if not success:
                            break
                        frame += 1
                        # image name
                        imgname = os.path.join(imgs_path,
                            'frame_%06d.jpg' % frame)
                        # save image
                        cv2.imwrite(imgname, image)

                # per frame
                cam_aa = cv2.Rodrigues(Rs[j])[0].T[0]
                pattern = os.path.join(imgs_path, '*.jpg')
                img_list = sorted(glob.glob(pattern))

                K, R, T = Ks[j][:3, :3], Rs[j], Ts[j]
                for i, img_i in enumerate(img_list[::10]):

                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    img_view = os.path.join('S' + str(user_i),
                                            'Seq' + str(seq_i),
                                            'imageFrames',
                                            'video_' + str(vid_i),
                                            img_name)
                    df_img_view = os.path.join('S' + str(user_i),
                                            'Seq' + str(seq_i),
                                            'imageFrames',
                                            'video_' + str(vid_i),
                                            'frame_' + img_name)
                    joints = np.reshape(annot2[vid_i][0][i], (28, 2))[joints17_idx]
                    S17 = np.reshape(annot3[vid_i][0][i], (28, 3))/1000
                    S17_world = np.matmul(R.T, S17.T).T - np.matmul(R.T, T)
                    S17 = S17[joints17_idx] - S17[4] # 4 is the root

                    # # reproject joints into the image
                    # projected_points = (K @ S17.T).T
                    # joints2d = projected_points[:, :2] / np.hstack((projected_points[:, 2:], projected_points[:, 2:]))
                    # img = cv2.imread(os.path.join(imgs_path, img_name))
                    # img_points = joints2d.astype(np.int)
                    # for pt in img_points:
                    #     cv2.circle(img, (int(pt[0]), int(pt[1])), radius=6, color=(0, 0, 255), thickness=-1)
                    # cv2.imshow('Vis', img)
                    # cv2.waitKey()

                    bbox = [min(joints[:,0]), min(joints[:,1]),
                            max(joints[:,0]), max(joints[:,1])]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200

                    # get SMPL Parameters
                    import ipdb; ipdb.set_trace()
                    theta = mpi_df[df_img_view == mpi_df['imgname']]['pose'][0]


                    # check that all joints are visible
                    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
                    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < len(joints_idx):
                        continue
                        
                    part = np.zeros([24,3])
                    part[joints_idx] = np.hstack([joints, np.ones([17,1])])
                    json_file = os.path.join(openpose_path, 'mpi_inf_3dhp',
                                             img_view.replace('.jpg', '_keypoints.json'))
                    openpose = read_openpose(json_file, part, 'mpi_inf_3dhp')

                    S = np.zeros([24,4])
                    S[joints_idx] = np.hstack([S17, np.ones([17,1])])
                    S_world = np.zeros([24, 4])
                    S_world[joints_idx] = np.hstack([S17_world, np.ones([17,1])])

                    # because of the dataset size, we only keep every 10th frame
                    counter += 1
                    if counter % 10 != 1:
                        continue

                    # store the data
                    imgnames_.append(img_view)
                    centers_.append(center)
                    scales_.append(scale)
                    parts_.append(part)
                    Ss_.append(S)
                    Ss_world_.append(S_world)
                    pose=poses_,
                    transl=transls_,
                    shape=shapes_,
                    openposes_.append(openpose)
                    pose_world=poses_world_,
                    transl_world=transls_world_,
                    cam_r=camr_r_,
                    cam_t=cam_t_,
                    cam_k=cam_k_,
                    in_bos_label=in_bos_label_,
                    contact_label=contact_label_,
                       
    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_inf_3dhp_train_world.npz')
    if fits_3d is not None:
        fits_3d = np.load(fits_3d)
        np.savez(out_file, imgname=imgnames_,
                           center=centers_,
                           scale=scales_,
                           part=parts_,
                           pose=fits_3d['pose'],
                           shape=fits_3d['shape'],
                           has_smpl=fits_3d['has_smpl'],
                           S=Ss_,
                           openpose=openposes_)
    else:
        np.savez(out_file, imgname=imgnames_,
                           center=centers_,
                           scale=scales_,
                           part=parts_,
                           S=Ss_,
                           openpose=openposes_)        
        
        
def test_data(dataset_path, out_path, joints_idx, scaleFactor, vis_path=None, visualize=None):

    joints17_idx = [14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]

    imgnames_, scales_, centers_, parts_,  Ss_ = [], [], [], [], []

    # training data
    user_list = range(1,7)

    for user_i in tqdm(user_list):
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])
        for frame_i, valid_i in enumerate(valid):
            if valid_i == 0:
                continue
            img_name = os.path.join('mpi_inf_3dhp_test_set',
                                   'TS' + str(user_i),
                                   'imageSequence',
                                   'img_' + str(frame_i+1).zfill(6) + '.jpg')

            joints = annot2[frame_i,0,joints17_idx,:]
            S17 = annot3[frame_i,0,joints17_idx,:]/1000
            S17 = S17 - S17[0]

            bbox = [min(joints[:,0]), min(joints[:,1]),
                    max(joints[:,0]), max(joints[:,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200

            # check that all joints are visible
            img_file = os.path.join(dataset_path, img_name)
            I = scipy.misc.imread(img_file)
            h, w, _ = I.shape
            x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
            y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)
            if np.sum(ok_pts) < len(joints_idx):
                continue

            part = np.zeros([24,3])
            part[joints_idx] = np.hstack([joints, np.ones([17,1])])

            S = np.zeros([24,4])
            S[joints_idx] = np.hstack([S17, np.ones([17,1])])

            # store the data
            imgnames_.append(img_name)
            centers_.append(center)
            scales_.append(scale)
            parts_.append(part)
            Ss_.append(S)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_inf_3dhp_test_world.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_)    

def mpi_inf_3dhp_extract(dataset_path, openpose_path, out_path, mode, extract_img=False, static_fits=None, vis_path=None, visualize=False):

    scaleFactor = 1.2
    joints_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]
    
    if static_fits is not None:
        fits_3d = os.path.join(static_fits, 
                               'mpi-inf-3dhp_mview_fits.npz')
    else:
        fits_3d = None
    
    if mode == 'train':
        train_data(dataset_path, openpose_path, out_path, 
                   joints_idx, scaleFactor, extract_img=extract_img, fits_3d=fits_3d, vis_path=vis_path, visualize=visualize)
    elif mode == 'test':
        test_data(dataset_path, out_path, joints_idx, scaleFactor, vis_path=vis_path, visualize=visualize)


# dataset_path = '/ps/project/datasets/mpi_inf_3dhp/'
# out_path = '/is/ps2/ppatel/SPIN_Finetuning2/spin_ft_on_agora/data/dataset_extras'
# mpi_inf_3dhp_extract(dataset_path,openpose_path=None,mode='train',out_path=out_path)
