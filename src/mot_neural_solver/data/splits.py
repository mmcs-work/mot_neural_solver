_SPLITS = {}
#################
# CTMCV1
#################
# sequences used for training
ctmcv1_train_seqs = ['3T3-run01']#['A-10-run01', '3T3-run01']#'3T3-run01','APM-run01','BPAE-run05','CV-1-run03','LLC-MK2-run07'] #, 'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Stadtmitte']
all_seqs = ['3T3-run01','A-10-run01'] #,'APM-run01','BPAE-run05','CV-1-run03','LLC-MK2-run07','MDBK-run09','MDOK-run09','PL1Ut-run01','U2O-S-run03','3T3-run03','A-10-run03','APM-run03','BPAE-run07','LLC-MK2-run01','MDBK-run01','MDOK-run01','OK-run01','PL1Ut-run03','U2O-S-run05','3T3-run05','A-10-run05','APM-run05','CRE-BAG2-run01','LLC-MK2-run02a','MDBK-run03','MDOK-run03','OK-run03','PL1Ut-run05','3T3-run07','A-10-run07','BPAE-run01','CRE-BAG2-run03','LLC-MK2-run03','MDBK-run05','MDOK-run05','OK-run05','RK-13-run01','3T3-run09','A-549-run03','BPAE-run03','CV-1-run01','LLC-MK2-run05','MDBK-run07','MDOK-run07','OK-run07','RK-13-run03']

# Additional train sequences not used for tranining (since they are present in MOT17 etc.)
add_ctmcv1_train_seqs = []
# _SPLITS['mot15_train_gt'] = {'2DMOT2015/train': [f'{seq}-GT' for seq in mot15_train_seqs]}
# _SPLITS['ctmcv1_train_gt'] = {'CTMCCVPR20/train': ctmcv1_train_seqs + add_ctmcv1_train_seqs}

# _SPLITS['ctmcv1_split_1_train_gt'] = {'CTMCCVPR20/train': ctmcv1_train_seqs}
# _SPLITS['split_1_ctmc_val'] = {'CTMCCVPR20/train': ['A-10-run01']}
# _SPLITS['ctmcv1_split_2_train_gt'] = {'CTMCCVPR20/train': ['A-10-run01']}
# _SPLITS['split_2_ctmc_val'] = {'CTMCCVPR20/train': ['3T3-run01']}

# This is used for debugging
#################################################################################
#_SPLITS['ctmcv1_split_1_train_gt'] = {'CTMCCVPR20/train': ['LLC-MK2-run07']}    #
#_SPLITS['split_1_ctmc_val'] = {'CTMCCVPR20/train': ['3T3-run01']}               #
#################################################################################
# _SPLITS['ctmcv1_split_1_train_gt'] = {'CTMCCVPR20/train': [f'{all_seqs[i]}' for i in range(len(all_seqs)) if i%2 == 0]}
# _SPLITS['split_1_ctmc_val'] = {'CTMCCVPR20/train': [f'{all_seqs[i]}' for i in range(len(all_seqs)) if i%2 == 1]}
# # _SPLITS['split_1_ctmc_val'] = {'CTMCCVPR20/train': [f'{all_seqs[i]}' for i in range(len(all_seqs)) if i%2 == 1]}
# _SPLITS['ctmcv1_split_2_train_gt'] = {'CTMCCVPR20/train': [f'{all_seqs[i]}' for i in range(len(all_seqs)) if i%2 == 1]}
# _SPLITS['split_2_ctmc_val'] = {'CTMCCVPR20/train': [f'{all_seqs[i]}' for i in range(len(all_seqs)) if i%2 == 0]}
# Test sequences
# test_seqs =  ['TUD-Crossing'] #, 'PETS09-S2L2', 'ETH-Jelmoli', 'ETH-Linthescher', 'ETH-Crossing', 'AVG-TownCentre',
#                 #   'ADL-Rundle-1', 'ADL-Rundle-3', 'KITTI-16', 'KITTI-19', 'Venice-1']
# _SPLITS['mot15_test'] = {'2DMOT2015/test': test_seqs}

# strat_over_framenum = [['MDBK-run09', 'MDOK-run09', 'LLC-MK2-run03', 'MDOK-run07', 'U2O-S-run05', 'U2O-S-run03', 'MDOK-run05', 'PL1Ut-run03', 'BPAE-run03', 'CRE-BAG2-run03', 'A-10-run05', 'APM-run03', 'OK-run01', 'MDBK-run01'], ['A-10-run07', '3T3-run03', '3T3-run07', 'CRE-BAG2-run01', 'LLC-MK2-run01', 'MDBK-run07', '3T3-run01', 'CV-1-run03', 'BPAE-run01', 'OK-run03', 'OK-run05'], ['BPAE-run07', '3T3-run09', '3T3-run05', 'A-10-run01', 'A-10-run03', 'LLC-MK2-run05', 'APM-run01', 'OK-run07', 'A-549-run03', 'RK-13-run01', 'MDBK-run05', 'RK-13-run03', 'MDBK-run03'], ['LLC-MK2-run07', 'MDOK-run01', 'APM-run05', 'CV-1-run01', 'BPAE-run05', 'PL1Ut-run01', 'PL1Ut-run05', 'LLC-MK2-run02a', 'MDOK-run03']]
# _SPLITS['ctmcv1_split_1_train_gt'] = {'CTMCCVPR20/train': strat_over_framenum[1]+strat_over_framenum[2]+strat_over_framenum[3]}
# _SPLITS['split_1_ctmc_val'] = {'CTMCCVPR20/train': strat_over_framenum[0]}
# _SPLITS['ctmcv1_split_2_train_gt'] = {'CTMCCVPR20/train': strat_over_framenum[0]+strat_over_framenum[2]+strat_over_framenum[3]}
# _SPLITS['split_2_ctmc_val'] = {'CTMCCVPR20/train': strat_over_framenum[1]}
# _SPLITS['ctmcv1_split_3_train_gt'] = {'CTMCCVPR20/train': strat_over_framenum[0]+strat_over_framenum[1]+strat_over_framenum[3]}
# _SPLITS['split_3_ctmc_val'] = {'CTMCCVPR20/train': strat_over_framenum[2]}
# _SPLITS['ctmcv1_split_4_train_gt'] = {'CTMCCVPR20/train': strat_over_framenum[0]+strat_over_framenum[1]+strat_over_framenum[2]}
# _SPLITS['split_4_ctmc_val'] = {'CTMCCVPR20/train': strat_over_framenum[3]}

# strat_over_framenum = [['A-10-run07', 'BPAE-run07', 'MDOK-run01', '3T3-run03', '3T3-run05', 'MDOK-run07', 'U2O-S-run05', 'U2O-S-run03', '3T3-run07', 'CRE-BAG2-run01', 'MDOK-run05', 'LLC-MK2-run01', 'PL1Ut-run03', 'MDBK-run07', 'APM-run01', '3T3-run01', 'CRE-BAG2-run03', 'OK-run07', 'A-549-run03', 'BPAE-run01', 'PL1Ut-run05', 'OK-run03', 'OK-run05', 'RK-13-run03', 'MDBK-run03', 'MDBK-run01'], ['MDBK-run09', 'LLC-MK2-run07', '3T3-run09', 'MDOK-run09', 'LLC-MK2-run03', 'APM-run05', 'A-10-run01', 'A-10-run03', 'LLC-MK2-run05', 'CV-1-run01', 'BPAE-run03', 'BPAE-run05', 'PL1Ut-run01', 'CV-1-run03', 'A-10-run05', 'RK-13-run01', 'MDBK-run05', 'APM-run03', 'OK-run01', 'LLC-MK2-run02a', 'MDOK-run03']]
# _SPLITS['ctmcv1_split_1_train_gt'] = {'CTMCCVPR20/train': strat_over_framenum[1]}
# _SPLITS['split_1_ctmc_val'] = {'CTMCCVPR20/train': strat_over_framenum[0]}
# _SPLITS['ctmcv1_split_2_train_gt'] = {'CTMCCVPR20/train': strat_over_framenum[0]}
# _SPLITS['split_2_ctmc_val'] = {'CTMCCVPR20/train': strat_over_framenum[1]}

strat_over_framenum = ['A-10-run07', 'BPAE-run07', 'MDOK-run01', '3T3-run03', '3T3-run05', 'MDOK-run07', 'U2O-S-run05', 'U2O-S-run03', '3T3-run07', 'CRE-BAG2-run01', 'MDOK-run05', 'LLC-MK2-run01', 'PL1Ut-run03', 'MDBK-run07', 'APM-run01', '3T3-run01', 'CRE-BAG2-run03', 'OK-run07', 'A-549-run03', 'BPAE-run01', 'PL1Ut-run05', 'OK-run03', 'OK-run05', 'RK-13-run03', 'MDBK-run03', 'MDBK-run01','MDBK-run09', 'LLC-MK2-run07', '3T3-run09', 'MDOK-run09', 'LLC-MK2-run03', 'APM-run05', 'A-10-run01', 'A-10-run03', 'LLC-MK2-run05', 'CV-1-run01', 'BPAE-run03', 'BPAE-run05', 'PL1Ut-run01', 'CV-1-run03', 'A-10-run05', 'RK-13-run01', 'MDBK-run05', 'APM-run03', 'OK-run01', 'LLC-MK2-run02a', 'MDOK-run03']
_SPLITS['ctmc_train_gt'] = {'CTMCV1/train': strat_over_framenum}
test_set = ['3T3-run02','3T3-run04','3T3-run06','3T3-run08','A-10-run02','A-10-run04','A-10-run06','A-549-run02','A-549-run04','APM-run02','APM-run04','APM-run06','BPAE-run02','BPAE-run04','BPAE-run06','CRE-BAG2-run02','CRE-BAG2-run04','CV-1-run02','CV-1-run04','LLC-MK2-run02b','LLC-MK2-run04','LLC-MK2-run06','MDBK-run02','MDBK-run04','MDBK-run06','MDBK-run08','MDBK-run10','MDOK-run02','MDOK-run04','MDOK-run06','MDOK-run08','OK-run02','OK-run04','OK-run06','PL1Ut-run02','PL1Ut-run04','RK-13-run02','U2O-S-run02','U2O-S-run04']
_SPLITS['ctmc_train_gt_test'] = {'CTMCV1/test': test_set}


#################
# MOT15
#################
dets = ('DPM', 'FRCNN', 'SDP')

# Train sequences:
mot15_train_seqs = ['KITTI-17' , 'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Stadtmitte']

# Additional train sequences not used for tranining (since they are present in MOT17 etc.)
add_mot15_train_seqs = ['ETH-Pedcross2', 'TUD-Campus', 'KITTI-13', 'Venice-2', 'ADL-Rundle-8', 'ADL-Rundle-6']
_SPLITS['mot15_train_gt'] = {'2DMOT2015/train': [f'{seq}-GT' for seq in mot15_train_seqs]}
_SPLITS['mot15_train'] = {'2DMOT2015/train': mot15_train_seqs + add_mot15_train_seqs}

# Test sequences
test_seqs =  ['TUD-Crossing' , 'PETS09-S2L2', 'ETH-Jelmoli', 'ETH-Linthescher', 'ETH-Crossing', 'AVG-TownCentre',
                 'ADL-Rundle-1', 'ADL-Rundle-3', 'KITTI-16', 'KITTI-19', 'Venice-1']
_SPLITS['mot15_test'] = {'2DMOT2015/test': test_seqs}


#################
# MOT17
#################
dets = ('DPM', 'FRCNN', 'SDP')

# Train sequences:
train_seq_nums=  (2, 4, 5, 9, 10, 11, 13)
_SPLITS['mot17_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in train_seq_nums]}
_SPLITS['mot17_train'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in train_seq_nums for det in dets]}
_SPLITS['mot17_train_sdp'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-SDP' for seq_num in train_seq_nums ]}


# Cross Validation splits
_SPLITS['mot17_split_1_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in [2]]}
_SPLITS['split_1_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in [4] for det in dets]}

# _SPLITS['mot17_split_1_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 5, 9, 10, 13)]}
# _SPLITS['split_1_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4,11) for det in dets]}

_SPLITS['mot17_split_2_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 11, 10, 13)]}
_SPLITS['split_2_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (5, 9) for det in dets]}

_SPLITS['mot17_split_3_train_gt'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (4, 5, 9, 11)]}
_SPLITS['split_3_val'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 10, 13) for det in dets]}

_SPLITS['debug'] = {'MOT17Labels/train': ['MOT17-02-FRCNN']}


# Test sequences
test_seq_nums=  (1, 3, 6, 7, 8, 12, 14)
_SPLITS['mot17_test'] = {'MOT17Labels/test': [f'MOT17-{seq_num:02}-{det}' for seq_num in test_seq_nums for det in dets]}

############
# MOT20
############

train_seq_nums=  (1, 2, 3, 5)
# Train / Val sequences
_SPLITS['mot20_train'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in train_seq_nums]}
_SPLITS['mot20_train_gt'] = {'MOT20/train': [f'MOT20-{seq_num:02}-GT' for seq_num in train_seq_nums]}
_SPLITS['mot20_train_wo_val'] = {'MOT20/train': [f'MOT20-{seq_num:02}-GT' for seq_num in (1, 2,  5)]}
_SPLITS['mot20_val'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (3,)]}

_SPLITS['mot20_train_gt+'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 9)],
                              'MOT20/train': [f'MOT20-{seq_num:02}-GT' for seq_num in train_seq_nums]}


# Cross-Val
for split_num, val_seq in enumerate(train_seq_nums, 1):
    _SPLITS[f'mot20_train_{split_num}'] = {'MOT17Labels/train': [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 9)],
                                'MOT20/train': [f'MOT20-{seq_num:02}-GT' for seq_num in train_seq_nums if seq_num != val_seq]}
    _SPLITS[f'mot20_val_{split_num}'] = {'MOT20/train': [f'MOT20-{val_seq:02}']}



# Test Sequences
_SPLITS['mot20_test'] = {'MOT20/test': [f'MOT20-{seq_num:02}' for seq_num in (4, 6, 7, 8)]}

# Combinations:
_SPLITS['all_train'] = {**_SPLITS['mot17_train_gt'], **_SPLITS['mot15_train_gt'], **_SPLITS['mot20_train_gt']}
_SPLITS['all_test'] = {**_SPLITS['mot17_test'], **_SPLITS['mot15_test'], **_SPLITS['mot20_test']}




