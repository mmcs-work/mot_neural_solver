wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh -b -p $HOME/anaconda3
source $HOME/anaconda3/bin/activate
conda env create -f environment.yaml
# conda env update -f environment.yaml
conda activate mot_neural_solver
rm -rf tracking_wo_bnw
git clone https://github.com/gbraso/tracking_wo_bnw.git
pip install -e tracking_wo_bnw
pip install -e .
bash scripts/setup/download_motcha.sh
bash scripts/setup/download_models.sh
bash scripts/setup/download_prepr_dets.sh
# bash scripts/setup/download_ctmcv1.sh
git config --global user.email "moinak.manna@gmail.com"
git config --global user.name "mainakmanna"
