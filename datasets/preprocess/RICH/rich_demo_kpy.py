### this toy demo script shows how to use RICH data in the subfolders

import cv2
import json
import numpy as np



with open('/ps/scratch/ps_shared/stripathi/4yogi/RICH/test/MultiIOI_201019_ID03581_parkingLot_Calibration06_Settings06_PushUp__2/data/00068/00/bbox_refine/00068_00.json', 'r') as f:
    bbox = json.load(f)

bbox_top_left = np.array([bbox['x1'], bbox['y1']])

with open('/ps/scratch/ps_shared/stripathi/4yogi/RICH/test/MultiIOI_201019_ID03581_parkingLot_Calibration06_Settings06_PushUp__2/data/00068/00/keypoints_refine/00068_00_keypoints.json', 'r') as f:
    keypoints = json.load(f)

keypoints = np.array(keypoints['people'][0]['pose_keypoints_2d']).reshape([-1, 3])

keypoints[:,:2] = keypoints[:,:2] + bbox_top_left

img = cv2.imread('/ps/scratch/ps_shared/stripathi/4yogi/RICH/test/MultiIOI_201019_ID03581_parkingLot_Calibration06_Settings06_PushUp__2/data/00068/00/images_orig/00068_00.bmp')

img[keypoints[:,1].astype(int), keypoints[:,0].astype(int), :] = [0, 255, 0] # green
cv2.circle(img, (int(bbox['x1']), int(bbox['y1'])), 4, (255, 0, 0), -1) #blue
cv2.circle(img, (int(bbox['x1_rect']), int(bbox['y1_rect'])), 4, (0, 0, 255), -1) #red

cv2.imshow('Vis', img)
cv2.waitKey()
##





