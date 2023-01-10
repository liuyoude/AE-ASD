# AE-ASD
Autoencoder (AE) based methods for anomalous sound detection (ASD). 

AE: [[DCASE2020 Task2 baseline]](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds)

IDNN: [[pdf]](https://arxiv.org/pdf/2005.09234.pdf)

## dataset
DCASE 2020 Task2 Dataset:
+ [[development dataset]](https://zenodo.org/record/3678171)
+ [[additional training dataset]](https://zenodo.org/record/3727685)
+ [[evaluation dataset]](https://zenodo.org/record/3841772)

The dataset path can be seen in `config.yaml`


## Run

```shell
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
```

## Results
### Result on development dataset 

 | Method / (AUC(%) / pAUC(%)) | ToyCar | ToyConveyor | Fan | Pump | Slider | Valve | Average |
 | --------     | :-----:| :----:  | :-----:| :----:  | :-----:| :----:  | :-----:|
 | AE           | 77.38(65.92)  | 67.88(57.52)  | 63.91(53.00) | 70.78(60.83) | 81.56(64.29)  | 55.57(50.77)  | 69.52(58.73)  |
 | VAE          | 75.00(63.59)  | 63.54(54.87)  | 60.03(51.45) | 70.42(59.35) | 76.62(60.79)  | 52.94(50.53)  | 66.43(56.77)  |
 | IDNN         | 78.91(68.99)  | 70.55(59.13)  | 65.69(54.19) | 71.75(60.10) | 84.10(65.94)  | 82.14(63.99)  | 75.44(62.06)  |
 
 ### Result on evaluation dataset 

 | Method / (AUC(%) / pAUC(%)) | ToyCar | ToyConveyor | Fan | Pump | Slider | Valve | Average |
 | --------     | :-----:| :----:  | :-----:| :----:  | :-----:| :----:  | :-----:|
 | AE           | 77.01(64.10)  | 82.19(62.76)  | 72.94(55.20) | 76.58(62.03) | 78.63(57.25)  | 49.32(49.83)  | 72.78(58.53)  |
 | VAE          | 75.87(63.02)  | 77.81(57.72)  | 68.82(54.22) | 73.02(61.00) | 75.32(55.88)  | 46.50(49.81)  | 69.56(56.94)  |
 | IDNN         | 81.57(70.16)  | 83.69(65.91)  | 75.42(57.41) | 79.29(62.27) | 81.79(58.00)  | 72.77(54.94)  | 79.09(61.45)  |