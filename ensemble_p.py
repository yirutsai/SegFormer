import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
# cand = ["mit-b4_800_9148","ensemble_9101_9133_9077_9113","mit-b4_9208","mit-b4_800_bs4"]          #923 best
cand = ["ensemble_9101_9133_9077_9113","mit-b4_9208","mit-b4_1000_9206","ensemble_9148_9152_9179"]
base_root = Path("./output/nvidia")

L = sorted(os.listdir(base_root/Path(cand[0])))
# output_dir = Path("ensemble_9148_9191_9208_9179_p9105")
output_dir = Path("ensemble_9191_9208_9206_9204_p9105")
os.makedirs(output_dir,exist_ok=True)
for file_name in L:
    ens = np.zeros((942,1716))
    # file_name = os.path.basename(file_name)
    for c in cand:
        ans = np.asarray(plt.imread(base_root/Path(c)/Path(file_name)))
        assert ans[:,:,0].all()==ans[:,:,1].all() ==ans[:,:,2].all()
        ans = ans[:,:,0]
        # ans = cv2.imread(str(base_root/Path(c)/Path(file_name)))
        ens +=ans
        # print(np.unique(ens))
    # precision = np.asarray(plt.imread(base_root/Path("mit-b5_800_9105")/Path(file_name)))
    precision = np.asarray(plt.imread(base_root/Path("best_nvidia_mit-b5_800_9105.pt")/Path(file_name)))
    assert precision[:,:,0].all()==precision[:,:,1].all() == precision[:,:,2].all()
    precision = precision[:,:,0]
    ens/=len(cand)
    # print(np.unique(ens))
    ens = np.where(ens>0.5,1,ens)
    ens = np.where(ens<0.5,0,ens)
    ens = np.where(ens==0.5,precision,ens)
    # print(np.unique(ens))
    plt.imsave(output_dir/Path(file_name),ens,cmap="gray")
    

