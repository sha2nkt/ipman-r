import numpy as np
import constants
from utils.geometry import perspective_projection
import torch

def get_2d_joints(joints_3d, pred_cam_t, focal_length,dev, batch_size ):
    rotation = torch.eye(3, device=dev).unsqueeze(0).expand(batch_size, -1, -1)
    camera_center = torch.zeros(batch_size, 2, device=dev)
    print('cam)t',pred_cam_t)
    print('rotaion',rotation)

    pred_keypoints_2d = perspective_projection(joints_3d,
                                               rotation=rotation,
                                               translation=pred_cam_t,
                                               focal_length=focal_length,
                                               camera_center=camera_center)

    return pred_keypoints_2d

def get_original(proc_parm, verts, cam, joints):
    img_size = proc_parm['img_size'][0]
    # undo_scale = np.array(proc_parm['scale'])#1. / np.array(proc_parm['scale'])
    undo_scale = img_size / constants.IMG_RES

    cam_pos = cam[1:]
    principal_pt = np.array([img_size, img_size]) / 2.
    flength = constants.FOCAL_LENGTH
    # tz = flength / (0.5 * constants.IMG_RES * cam_s)
    tz = (2 * flength) / (constants.IMG_RES * cam[0] + 1e-9)
    trans = np.hstack([cam_pos, tz])
    vert_shifted = (verts + trans)

    start_pt = proc_parm['start_pt']
    final_principal_pt = principal_pt+ start_pt
    cam_for_render = np.hstack(
        [np.mean(flength * undo_scale), final_principal_pt[1], final_principal_pt[0]])

    # This is in padded image.
    # kp_original = (joints + proc_param['start_pt']) * undo_scale
    # Subtract padding from joints.
    # import ipdb; ipdb.set_trace()
    # kp_original = (joints + proc_parm['start_pt']) * undo_scale
    kp_original = (joints) * undo_scale + proc_parm['start_pt'][::-1]- (proc_parm['pad_x'][0],proc_parm['pad_y'][0])

    return cam_for_render, vert_shifted, kp_original

# def get_original(proc_parm, verts, cam, joints):
#
#     img_size = np.array(224)
#     undo_scale = 1.
#     print('und scale',undo_scale)
#    # undo_scale = img_size / constants.IMG_RES
#
#
#     cam_pos = cam[1:]
#     principal_pt = np.array([img_size, img_size]) / 2.
#     flength = 500
#     # tz = flength / (0.5 * constants.IMG_RES * cam_s)
#     tz = (2 * flength) / (constants.IMG_RES * cam[ 0] + 1e-9)
#     trans = np.hstack([cam_pos, tz])
#     vert_shifted = (verts + trans)
#     print('img size', img_size)
#     start_pt = proc_parm['start_pt']
#     print('start pt', proc_parm['start_pt'] )
#     print('principal pt', principal_pt)
#     final_principal_pt = (principal_pt  + start_pt)*undo_scale
#     cam_for_render = np.hstack(
#         [np.mean(flength * undo_scale), 1280/2,720/2])
#
#     # This is in padded image.
#     # kp_original = (joints + proc_param['start_pt']) * undo_scale
#     # Subtract padding from joints.
#     # import ipdb; ipdb.set_trace()
#     # kp_original = (joints + proc_parm['start_pt']) * undo_scale
#
#     kp_original = (joints ) * undo_scale + proc_parm['start_pt'][::-1] - (proc_parm['pad_x'][0],proc_parm['pad_y'][0])
#
#     return cam_for_render, vert_shifted, kp_original
