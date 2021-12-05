RUN_ID=ctmc_cv
python scripts/train.py with run_id=$RUN_ID cross_val_split=1
python scripts/train.py with run_id=$RUN_ID cross_val_split=2
# python scripts/train.py with run_id=$RUN_ID cross_val_split=3
# python scripts/train.py with run_id=$RUN_ID cross_val_split=4
# python scripts/train.py with run_id=$RUN_ID cross_val_split=3
python scripts/cross_validation.py with run_id=$RUN_ID