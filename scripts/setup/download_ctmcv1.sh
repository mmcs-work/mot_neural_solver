DATA_PATH=$(python -c "from mot_neural_solver.path_cfg import DATA_PATH; print(DATA_PATH)")
wget -P $DATA_PATH https://motchallenge.net/data/CTMCV1.zip
wget -P $DATA_PATH https://motchallenge.net/data/CTMCV1Labels.zip

# wget -P $DATA_PATH https://motchallenge.net/data/2DMOT2015.zip
# wget -P $DATA_PATH https://motchallenge.net/data/MOT17Det.zip
# wget -P $DATA_PATH https://motchallenge.net/data/MOT17Labels.zip

unzip -d $DATA_PATH $DATA_PATH/CTMCV1.zip
unzip -d $DATA_PATH/CTMCV1Labels $DATA_PATH/CTMCV1Labels.zip


# unzip -d $DATA_PATH $DATA_PATH/2DMOT2015.zip
# unzip -d $DATA_PATH/MOT17Labels $DATA_PATH/MOT17Labels.zip
# unzip -d $DATA_PATH/MOT17Det $DATA_PATH/MOT17Det.zip
# rm $DATA_PATH/{MOT17Labels,MOT17Det,2DMOT2015}.zip

rm $DATA_PATH/{CTMCV1Labels,CTMCV1}.zip
