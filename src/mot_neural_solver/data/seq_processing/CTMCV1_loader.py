import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

import cv2

from mot_neural_solver.path_cfg import DATA_PATH

import os
import os.path as osp

from mot_neural_solver.utils.iou import iou

import shutil

# FPS per seq are retrieved from https://motchallenge.net/data/2D_MOT_2015/
FPS_DICT = {
    '3T3-run01':30,
    'A-10-run01':30,
    'APM-run01':30,
    'BPAE-run05':30,
    'CV-1-run03':30,
    'LLC-MK2-run07':30,
    'MDBK-run09':30,
    'MDOK-run09':30,
    'PL1Ut-run01':30,
    'U2O-S-run03':30,
    '3T3-run03':30,
    'A-10-run03':30,
    'APM-run03':30,
    'BPAE-run07':30,
    'LLC-MK2-run01':30,
    'MDBK-run01':30,
    'MDOK-run01':30,
    'OK-run01':30,
    'PL1Ut-run03':30,
    'U2O-S-run05':30,
    '3T3-run05':30,
    'A-10-run05':30,
    'APM-run05':30,
    'CRE-BAG2-run01':30,
    'LLC-MK2-run02a':30,
    'MDBK-run03':30,
    'MDOK-run03':30,
    'OK-run03':30,
    'PL1Ut-run05':30,
    '3T3-run07':30,
    'A-10-run07':30,
    'BPAE-run01':30,
    'CRE-BAG2-run03':30,
    'LLC-MK2-run03':30,
    'MDBK-run05':30,
    'MDOK-run05':30,
    'OK-run05':30,
    'RK-13-run01':30,
    '3T3-run09':30,
    'A-549-run03':30,
    'BPAE-run03':30,
    'CV-1-run01':30,
    'LLC-MK2-run05':30,
    'MDBK-run07':30,
    'MDOK-run07':30,
    'OK-run07':30,
    'RK-13-run03':30
    }

MOV_CAMERA_DICT  =    { 
    '3T3-run01':False,
    'A-10-run01':False,
    'APM-run01':False,
    'BPAE-run05':False,
    'CV-1-run03':False,
    'LLC-MK2-run07':False,
    'MDBK-run09':False,
    'MDOK-run09':False,
    'PL1Ut-run01':False,
    'U2O-S-run03':False,
    '3T3-run03':False,
    'A-10-run03':False,
    'APM-run03':False,
    'BPAE-run07':False,
    'LLC-MK2-run01':False,
    'MDBK-run01':False,
    'MDOK-run01':False,
    'OK-run01':False,
    'PL1Ut-run03':False,
    'U2O-S-run05':False,
    '3T3-run05':False,
    'A-10-run05':False,
    'APM-run05':False,
    'CRE-BAG2-run01':False,
    'LLC-MK2-run02a':False,
    'MDBK-run03':False,
    'MDOK-run03':False,
    'OK-run03':False,
    'PL1Ut-run05':False,
    '3T3-run07':False,
    'A-10-run07':False,
    'BPAE-run01':False,
    'CRE-BAG2-run03':False,
    'LLC-MK2-run03':False,
    'MDBK-run05':False,
    'MDOK-run05':False,
    'OK-run05':False,
    'RK-13-run01':False,
    '3T3-run09':False,
    'A-549-run03':False,
    'BPAE-run03':False,
    'CV-1-run01':False,
    'LLC-MK2-run05':False,
    'MDBK-run07':False,
    'MDOK-run07':False,
    'OK-run07':False,
    'RK-13-run03':False
}


#DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')

def _build_seq_info_dict_ctmcv1(seq_name, data_root_path, dataset_params):

    seq_path = osp.join(data_root_path, seq_name)
    imgs_path = osp.join(seq_path, 'img1')
    seq_len = len(set(os.listdir(imgs_path)))
    frame_height, frame_width, _= cv2.imread(osp.join(imgs_path, '000001.jpg')).shape

    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': dataset_params['det_file_name'],

                     'frame_height': frame_height,
                     'frame_width': frame_width,

                     'seq_len': seq_len,
                     'fps': FPS_DICT[seq_name],
                    
                     'mov_camera': MOV_CAMERA_DICT[seq_name],
                     'has_gt': osp.exists(osp.join(data_root_path, seq_name, 'gt'))}
    return seq_info_dict

# def get_mot15_det_df(seq_name, data_root_path, dataset_params):

#     seq_path = osp.join(data_root_path, seq_name)
#     detections_file_path = osp.join(seq_path, f"det/{dataset_params['det_file_name']}.txt")
#     det_df = pd.read_csv(detections_file_path, header=None)

#     # Number and order of columns is always assumed to be the same
#     det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]].copy()
#     det_df.columns = DET_COL_NAMES
#     det_df['bb_left'] -= 1 # Coordinates are 1 based
#     det_df['bb_top'] -= 1

#     # If id already contains an assignment (e.g. using tracktor output), keep it
#     if len(det_df['id'].unique()) > 1:
#         det_df['tracktor_id'] = det_df['id']

#     # Add each frame's path
#     add_frame_path = lambda frame_num: osp.join(data_root_path, seq_name, f'img1/{frame_num:06}.jpg')
#     det_df['frame_path'] = det_df['frame'].apply(add_frame_path)

#     seq_info_dict = _build_seq_info_dict_mot15(seq_name, data_root_path, dataset_params)
#     seq_info_dict['is_gt'] = False

#     if seq_info_dict['has_gt']: # Return the corresponding ground truth, if available
#         gt_file_path = osp.join(seq_path, f"gt/gt.txt")
#         gt_df = pd.read_csv(gt_file_path, header=None)
#         gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
#         gt_df.columns = GT_COL_NAMES
#         gt_df['bb_left'] -= 1  # Coordinates are 1 based
#         gt_df['bb_top'] -= 1
#         gt_df = gt_df[gt_df['conf'] == 1].copy()
#         gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
#         gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

#         # Store the gt file in the common evaluation path
#         gt_to_eval_path = osp.join(DATA_PATH, 'MOT_eval_gt', seq_name, 'gt')
#         os.makedirs(gt_to_eval_path, exist_ok=True)
#         shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

#     else:
#         gt_df = None

#     return det_df, seq_info_dict, gt_df

def get_ctmcv1_det_df_from_gt(seq_name, data_root_path, dataset_params):

    # Create a dir to store Ground truth data in case if does not exist yet
    seq_path = osp.join(data_root_path, seq_name)
    if not osp.exists(seq_path):
        os.mkdir(seq_path)
        non_gt_seq_path = osp.join(data_root_path, seq_name)
        shutil.copytree(osp.join(non_gt_seq_path, 'gt'), osp.join(seq_path, 'gt'))

    detections_file_path = osp.join(data_root_path, seq_name, f"gt/gt.txt")
    det_df = pd.read_csv(detections_file_path, header=None)

    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(GT_COL_NAMES)]]
    det_df.columns = GT_COL_NAMES
    det_df['bb_left'] -= 1 # Coordinates are 1 based
    det_df['bb_top'] -= 1

    # VERY IMPORTANT: Only take active annotations (see: https://arxiv.org/abs/1504.01942, page 7)
    det_df = det_df[det_df['conf'] == 1].copy()

    det_df['bb_bot'] = (det_df['bb_top'] + det_df['bb_height']).values
    det_df['bb_right'] = (det_df['bb_left'] + det_df['bb_width']).values
    det_df['bb_size'] = det_df['bb_height']*det_df['bb_width']

    det_df = drop_occluded_gt_annotations(det_df, dataset_params)

    # Add each image's path
    add_frame_path = lambda frame_num: osp.join(data_root_path, seq_name, f'img1/{frame_num:06}.jpg')
    det_df['frame_path'] = det_df['frame'].apply(add_frame_path)

    seq_info_dict = _build_seq_info_dict_ctmcv1(seq_name, data_root_path, dataset_params)

    # Correct the detections file name to contain the 'gt' as well as other attributes
    seq_info_dict['det_file_name'] = 'gt'
    # seq_info_dict['seq_path'] += '-GT'
    # seq_info_dict['seq'] += '-GT'
    seq_info_dict['is_gt'] = True

    # Store the gt file in the common evaluation path
    gt_file_path = osp.join(seq_path, f"gt/gt.txt")
    gt_to_eval_path = osp.join(DATA_PATH, 'MOT_eval_gt', seq_name, 'gt')
    os.makedirs(gt_to_eval_path, exist_ok=True)
    shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

    return det_df, seq_info_dict, None

####################################################################################################################
# (Messy) Functions used to process MOT15 GT boxes and use them for training
####################################################################################################################

def drop_occluded_gt_annotations(gt_df, dataset_params):
    """
    Unlike MOT17 ground truth boxes, MOT15 boxes do not have a visibility score. Using all (even fully occluded) boxes
    for training would provide a 'confusing signal' for learning + would show situations that would not be seen at test
    time (detectors + nms will not give boxes for occluded pedestrians).

    In this function we attempt to heuristically identify occluded boxes.

    Heuristics:
    - First drop all bounding boxes that almost completely overlap with others (high IoU). Since we
    cannot know for sure which one occludes the other one, we just drop both of them.
    - Then apply (essentially) NMS replacing IoU with a modified version where we compute intersect / smallest area. This
    is essentially a 'containment' score among boxes

    The second step allows us to identify situations in which a (smaller) target is (almost) completely occluded
    by a larger one, even though IoU might be small.

    """
    # First get the indices of pairs of boxes that should not be compared:
    time_dist_matrix = gt_df.frame.values.reshape(-1, 1) - gt_df.frame.values.reshape(1, -1)
    not_same_frame_ixs = np.where(time_dist_matrix != 0)
    same_detect_ixs = np.diag_indices(gt_df.shape[0])
    invalid_ixs = tuple([np.concatenate((not_same_frame_ixs[i], same_detect_ixs[i])) for i in [0, 1]])

    # Compute IoU and set it to zero for invalid pairs
    iou_matrix = iou(gt_df[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values,
                     gt_df[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values)
    iou_matrix[invalid_ixs] = 0
    drop_step_1 = iou_matrix.max(axis=0) > dataset_params['GT_train_max_iou_thresh']

    # Now, compute the 'modified' IoU
    occluded_matrix = intersec_over_min_max_area(gt_df[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values,
                                                 gt_df[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values,
                                                 denom_operator='min')

    # Determine, for each pair of bounding boxes, which one has the smallest area
    smaller_area = gt_df.bb_size.values.reshape(1, -1) < gt_df.bb_size.values.reshape(-1, 1)  # True if
                                                                                              # row < column??
    smaller_area[invalid_ixs] = False
    occluded_matrix[invalid_ixs] = 0
    drop_step_2 = ((occluded_matrix > dataset_params['GT_train_max_iou_containment_thresh'])
                   & smaller_area).max(axis=0)

    gt_df = gt_df[(~drop_step_1) & (~drop_step_2)].copy()

    return gt_df

def intersec_over_min_max_area(bboxes1, bboxes2, denom_operator = 'min'):
    """
    Vectorized version of IoU, over two lists of bounding boxes, each of them given by their upper left and lower
    right vertex coordinates

    Modified version to compute, instead of the ratio : a(intersction(b1, b2)) / a(union(b1, b2)),
    the ratio a(intersection(b1, b2)) / min_max(a(b1), a(b2))

    In case min_max is a min operation, this ratio is 1 when one box is contained inside the other.

    """
    operator_dict = {'min': np.minimum, 'max': np.maximum}
    assert denom_operator in operator_dict
    operator  = operator_dict[denom_operator]


    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    # x11, y11, x12, y12 = bboxes1

    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    # print(yB.shape)
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / operator(boxAArea, np.transpose(boxBArea))

    return iou



