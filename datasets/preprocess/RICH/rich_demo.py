### this toy demo script shows how to use RICH data in the subfolders

import os
import cv2
import json
import numpy as np
import trimesh
import os.path as osp

def extract_cam_param_xml(xml_path:str='', dtype=np.float):
    
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
    distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = np.array([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    
    rotation = np.array([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]], 
                            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]], 
                            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = np.array([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
                    -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11], 
                    -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

    cam_center =np.array([cam_center], dtype=dtype)

    k1 = np.array([distortion_vec[0]], dtype=dtype)
    k2 = np.array([distortion_vec[1]], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2


################## meta data #################
SPLIT = 'val'
SEQ_NAME ='2021-06-15_Multi_IOI_ID_03588_Yoga1'
ROOT_DIR = '/ps/scratch/ps_shared/stripathi/4yogi/RICH'
cam_id = 2
seq_dir = osp.join(ROOT_DIR,SPLIT,SEQ_NAME)
frame_id = 100 ## randomly chosen frame


################## visualize body in scan coordidate #################

## body resides in multi-ioi coordidate, where camera 0 is world zero.
body_mesh = trimesh.load(os.path.join(seq_dir,'params', f'{frame_id:05d}', '00/meshes/000.ply'), process=False)

## ground resides in Leica scan coordidate, which is (roughly) axis aligned. 
ground_mesh = trimesh.load(os.path.join(seq_dir,'ground_mesh.ply'), process=False)

## rigid transformation between multi-ioi and Leica scan
ioi2scan_fn = os.path.join(seq_dir,'cam2scan.json')
with open(ioi2scan_fn, 'r') as f:
    ioi2scan_dict = json.load(f)
    R = np.array(ioi2scan_dict['R'])
    t = np.array(ioi2scan_dict['t']).reshape(1, 3)

vertices_scan = body_mesh.vertices @ R + t # row vector --> right multiplication
body_mesh.vertices = vertices_scan

ground_eq = np.mean(ground_mesh.vertices, axis=0)
print(f'Ground plane equation: z = {ground_eq[2]}')
(body_mesh + ground_mesh).show()



################## visualize body kpts in images #################

## load the body (in ioi coordinate) again
body_mesh = trimesh.load(os.path.join(seq_dir,'params', f'{frame_id:05d}', '00/meshes/000.ply'), process=False)

## load and build camera params
camera_fn = os.path.join(seq_dir,f'calibration/{cam_id:03d}.xml')
focal_length_x, focal_length_y, center, rotation, translation, _, _, _ = extract_cam_param_xml(camera_fn)

K = np.eye(3, dtype=np.float)
K[0, 0] = focal_length_x
K[1, 1] = focal_length_y
K[:2, 2:] = center.T

## transform to each camera coordinate and project to the image domain
vertices_cam = body_mesh.vertices @ rotation.T + translation
projected_points = (K @ vertices_cam.T).T
img_points = projected_points[:, :2] / np.hstack((projected_points[:, 2:], projected_points[:, 2:]))

## visualize projected points
img = cv2.imread(osp.join(seq_dir,f'data/{frame_id:05d}','00/images_orig', f'{frame_id:05d}_{cam_id:02d}.png'))
img_points = img_points.astype(np.int)
img[img_points[:,1], img_points[:,0], :] = [0, 255, 0]
cv2.imshow('Vis', img)
cv2.waitKey()
##





