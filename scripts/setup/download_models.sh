#!/usr/bin/env bash

OUTPUT_PATH=$(python -c "from mot_neural_solver.path_cfg import OUTPUT_PATH; print(OUTPUT_PATH)")
wget -P $OUTPUT_PATH/trained_models/reid https://vision.in.tum.de/webshare/u/brasoand/mot_neural_solver/resnet50_market_cuhk_duke.tar-232
wget -P $OUTPUT_PATH/trained_models/frcnn https://vision.in.tum.de/webshare/u/brasoand/mot_neural_solver/frcnn_epoch_27.pt.tar
wget -P $OUTPUT_PATH/trained_models/graph_nets https://vision.in.tum.de/webshare/u/brasoand/mot_neural_solver/mot_mpnet_epoch_006.ckpt

cp /content/drive/MyDrive/output/models/model-best.pth.tar $OUTPUT_PATH/trained_models/reid/resnet50_ctmc.pth.tar
cp /content/drive/MyDrive/output/models/model-best-split-1.pth.tar $OUTPUT_PATH/trained_models/reid/resnet50_ctmc_1.pth.tar
cp /content/drive/MyDrive/output/models/reid-full.pth.tar $OUTPUT_PATH/trained_models/reid/resnet50_ctmc_full.pth.tar
