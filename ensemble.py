import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
cand = ["mit-b4_9101","mit-b5_9077","mit-b4_9208","mit-b5_9133"]
base_root = Path("./output/nvidia")

L = sorted(os.listdir(base_root/Path(cand[0])))
output_dir = Path("ensemble_9101_9077_9208_9133")
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
    ens/=len(cand)
    # print(np.unique(ens))
    ens = np.where(ens>=0.5,1,0)
    # print(np.unique(ens))
    plt.imsave(output_dir/Path(file_name),ens,cmap="gray")
    

