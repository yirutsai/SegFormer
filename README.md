<!-- sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y -->

# DLMI Task2
## Installation
<!-- pip install opencv-python
pip install albumentations
pip install datasets
pip install transformers
pip install wandb -->
```
conda env create -f environment.yml
conda activate DLMI_Task2
```

## Preprocess
```
python preprocess.py
python divide.py
```

## Training
```
python final0518.py \
        --lr <learning_rate> \
        --model_type <pretrained_weight> \
        --bs <batch_size> \
        --n_epochs <n_epochs> \
        --workers <num_workers for dataloader>\
        --patience <patience>\
        --img <img_size>\
        --seed <seed>\
        [--do_predict]
```

If `do_predict` is launched, it will only inference and no training. If not, it will still inference after training is done.

Sample:
```
python final0518.py --lr 3e-5 --model_type nvidia/mit-b4 --bs 1 --img 800 --patience 150 --n_epochs 200
```


