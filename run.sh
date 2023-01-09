# set tsinghua source
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# install package
pip install -r requirements.txt
# run
## AE
python run.py
## VAE
python run.py --vae=true
## IDNN
python run.py --idnn=true

# get evaluation dataset(DCASE 2020 Task2) result
cd ./evaluator
python evaluator.py

# visualization
tensorboard --logdir=runs