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
## Ensemble
In order to save GPU memory, we ensemble the output of the model.
You should inference before ensembling.
You should adjust the cand in line #7 to choose which directory to ensemble. And change the `output_dir` in line 11 to rename the output_dir.
```
python ensemble.py
```

## Reproduce best performance
1. Ensemble `best_nvidia_mit-b4_9101.pt`,`best_nvidia_mit-b5_9133.pt`,`best_nvidia_mit-b5_9077.pt`,`best_nvidia_mit-b5_9133.pt` to generate the first candidate.
2. Ensemble `best_nvidia_mit-b4_800_9148.pt`,`best_nvidia_mit-b4_800_bs4_9179.pt`,`best_nvidia_mit-b5_800_9152.pt` to generate the second candidate.
3. Ensemble the first and second candidate with `best_nvidia_mit-b4_1000_9206.pt`(,`best_nvidia_mit-b4_840_9208.pt` there should be this file, however it corrupted accidently)


