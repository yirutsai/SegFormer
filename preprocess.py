from __future__ import annotations
import json
import cv2
import os
import PIL.Image
import numpy as np
from pathlib import Path
from labelme import utils
mask_root =Path(os.path.dirname(__file__))/Path("SEG_Train_Datasets/Train_Masks")
os.makedirs(mask_root,exist_ok=True)
data_root = Path(os.path.dirname(__file__))/Path("SEG_Train_Datasets/Train_Annotations")
print(len(os.listdir(data_root)))
for file_name in sorted(os.listdir(data_root)):
    with open(data_root/file_name,"r") as f:
        data = json.load(f)
        h,w = data["imageHeight"], data["imageWidth"]
        lbl,lbl_names = utils.shape.labelme_shapes_to_label((h,w),data["shapes"])
        lbl *=255
        # PIL.Image.fromarray(lbl).save(str(mask_root/file_name).split(".")[0]+".png")
        cv2.imwrite(str(mask_root/file_name).split(".")[0]+".png",lbl)
        image = cv2.imread(str(mask_root/file_name).split(".")[0]+".png")

