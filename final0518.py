
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import cv2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as albu
from argparse import ArgumentParser
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_metric

from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation

parser = ArgumentParser()
parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--model_type",type = str,default="nvidia/mit-b5")
parser.add_argument("--bs",type=int,default=2)
parser.add_argument("--workers",type=int,default=0)
parser.add_argument("--n_epochs",type=int,default=100)
parser.add_argument("--do_predict",action="store_true")
parser.add_argument("--patience",type=int,default=20)
parser.add_argument("--img",type=int,default=512)
parser.add_argument("--feature_ext_type",type=str,default="nvidia/mit-b5")
parser.add_argument("--seed",type=int,default=3030)

args = parser.parse_args()
myseed = args.seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True,inference = False,aug = False):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train
        self.inference = inference
        self.aug = aug
        if(aug):
            self.augmentation = self.get_training_augmentation()
        if(inference==False):
            sub_path = "train" if self.train else "valid"
            self.img_dir = os.path.join(self.root_dir, sub_path, "Train_Images")
            self.ann_dir = os.path.join(self.root_dir, sub_path, "Train_Masks")
        else:
            self.img_dir = os.path.join(self.root_dir)
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)
        
        if(inference==False):
            # read annotations
            annotation_file_names = []
            for root, dirs, files in os.walk(self.ann_dir):
                annotation_file_names.extend(files)
            self.annotations = sorted(annotation_file_names)

            assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)
    def get_training_augmentation(self):
        train_transform = [

            albu.HorizontalFlip(p=0.5),
            albu.Rotate(limit=40,p=1,border_mode=cv2.BORDER_CONSTANT),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.8, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            
            albu.HueSaturationValue(p=0.3),
            albu.Sharpen(p=0.3),
            albu.RandomBrightnessContrast(p=0.3),

            albu.Crop(x_min=0, y_min=0, x_max=800, y_max=750, p=0.5),
            albu.PadIfNeeded(800, 800)

            
        ]
        return albu.Compose(train_transform)
    def __getitem__(self, idx):
        
        image = np.asarray(Image.open(os.path.join(self.img_dir, self.images[idx])))
        if(self.inference==False):
            segmentation_map = np.asarray(Image.open(os.path.join(self.ann_dir, self.annotations[idx])))
            # print(np.max(segmentation_map))
            # print(np.unique(segmentation_map))
            segmentation_map[segmentation_map==255] =1
            # print(np.max(segmentation_map))
        else:
            segmentation_map = np.zeros_like(image[:,:,0])
        # print(np.max(segmentation_map))
        # randomly crop + pad both image and segmentation map to same size
        if(self.aug):
            sample = self.augmentation(image=image, mask=segmentation_map)
            image, segmentation_map = sample['image'], sample['mask']
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs

root_dir = 'Images_Train90Valid10'
feature_extractor = SegformerFeatureExtractor.from_pretrained(args.feature_ext_type,size=args.img)

train_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor,aug=True)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, train=False)
public_dataset = SemanticSegmentationDataset(root_dir="Public_Image", feature_extractor=feature_extractor, train=False, inference = True)
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(valid_dataset))
print("Number of public examples:", len(public_dataset))



train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,num_workers=args.workers)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.bs,num_workers=args.workers)
public_dataloader = DataLoader(public_dataset,batch_size=1,shuffle=False)
# public_dataloader = DataLoader(valid_dataset, batch_size=1)



# load id2label mapping from a JSON on the hub
id2label = {0:"bg",1:"stas"}
label2id = {v: k for k, v in id2label.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# define model
if(args.do_predict==False):
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_type,
                                                            num_labels=2, 
                                                            id2label=id2label, 
                                                            label2id=label2id,
    )
    # model = SegformerForSemanticSegmentation.from_pretrained(args.model_type)
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # move model to GPU
    
    model.to(device)

    best_iou = 0
    model.train()
    patience = args.patience
    ct = 0


metric = load_metric("mean_iou")





if(args.do_predict):
    n_epochs = 0
else:
    n_epochs = args.n_epochs
    import wandb
    wandb.init(project="Task2", entity="yirutsai",name=args.model_type,config=args)

for epoch in range(n_epochs):  # loop over the dataset multiple times
    metric = load_metric("mean_iou")
    print(f"Epoch:{epoch}/{n_epochs}")
    train_losses = []
    for idx, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    eval_losses = []
    for idx, batch in tqdm(enumerate(valid_dataloader),total = len(valid_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        # evaluate
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values,labels=labels)
            loss ,logits = outputs.loss,outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)
            eval_losses.append(loss.item())
            # note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # let's print loss and metrics every 100 batches

    metrics = metric.compute(num_labels=len(id2label), 
                            ignore_index=255,
                           reduce_labels=False, # we've already reduced the labels before)
    )

    print(f"Train Loss: {sum(train_losses)/len(train_losses)}")
    print(f"Eval loss: {sum(eval_losses)/len(eval_losses)}")
    print("Mean_iou:", metrics["mean_iou"])
    print("Mean accuracy:", metrics["mean_accuracy"])
    wandb.log({"loss":loss.item(),"mean_iou":metrics["mean_iou"],"mean_acc":metrics["mean_accuracy"]})
    # wandb.watch(model)
    if(metrics["mean_iou"]>=best_iou):
        best_iou = metrics["mean_iou"]
        torch.save(model,f"best_{args.model_type.replace('/','_')}.pt")
        print(f"Model saved as best_{args.model_type.replace('/','_')}.pt")
        ct = 0
    else:
        ct +=1
        if(ct>patience):
            break
if(args.do_predict==False):
    del model
model = torch.load(f"best_{args.model_type.replace('/','_')}.pt")
# import ttach as tta
# model = tta.SegmentationTTAWrapper(model,tta.aliases.d4_transform(),merge_mode='mean')

import matplotlib.pyplot as plt
os.makedirs(f"output/{args.model_type}",exist_ok=True)
for idx,batch in enumerate(public_dataloader):
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)
    # print(labels.shape,torch.max(labels),torch.unique(labels))
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values,labels=labels)
        # outputs = model(pixel_values,labels)
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(logits, size=(942,1716), mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1).squeeze().detach().cpu().numpy()
    # print()
    # upsampled_image = nn.functional.interpolate(pixel_values, size=(942,1716), mode="bilinear", align_corners=False).squeeze().permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    # upsampled_labels = nn.functional.interpolate(labels.unsqueeze(0).float(), size=(942,1716), mode="bilinear", align_corners=False).squeeze().detach().cpu().numpy().astype(np.uint8)
    # print(np.max(upsampled_image))
    # print(predicted.shape)
    # print(np.max(predicted))
    # print(upsampled_image.shape)
    # plt.imsave(f"B5_output/labels_{valid_dataset.images[idx].replace('.jpg','.png')}",upsampled_labels,cmap="gray")
    # plt.imsave(f"B5_output/ori_{valid_dataset.images[idx].replace('.jpg','.png')}",upsampled_image,cmap="gray")
    print(f"output/{args.model_type}/{public_dataset.images[idx].replace('.jpg','.png')}")
    plt.imsave(f"output/{args.model_type}/{public_dataset.images[idx].replace('.jpg','.png')}",predicted,cmap="gray")
    # exit()



