import glob
from pathlib import Path
import shutil
import os
listdir = ['3T3-run02','3T3-run04','3T3-run06','3T3-run08','A-10-run02','A-10-run04','A-10-run06','A-549-run02','A-549-run04','APM-run02','APM-run04','APM-run06','BPAE-run02','BPAE-run04','BPAE-run06','CRE-BAG2-run02','CRE-BAG2-run04','CV-1-run02','CV-1-run04','LLC-MK2-run02b','LLC-MK2-run04','LLC-MK2-run06','MDBK-run02','MDBK-run04','MDBK-run06','MDBK-run08','MDBK-run10','MDOK-run02','MDOK-run04','MDOK-run06','MDOK-run08','OK-run02','OK-run04','OK-run06','PL1Ut-run02','PL1Ut-run04','RK-13-run02','testseq.txt','U2O-S-run02','U2O-S-run04']

for f in glob.glob('./testset/cont_det/*.txt'):
    dir = Path(f).stem
    print(f)
    if dir in listdir:
        os.makedirs(os.path.dirname(f'./data/CTMCCVPR20/test/{dir}/det/tracktor_prepr_det.txt'), exist_ok=True)
        shutil.copyfile(f,f'./data/CTMCCVPR20/test/{dir}/det/tracktor_prepr_det.txt')