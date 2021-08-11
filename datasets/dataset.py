from typing import List
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch
import random
from utils import Params
import albumentations as albu
from generators.augmentations import readIm,crop_pad,random_darkening


from PIL import Image

params=Params("./params.json")

class SegmentationDataset(Dataset):
    def __init__(
        self,
        images: List[Path],
        num_classes: int,
        image_size: int, #only squares right now 
        masks: List[Path] = None,
        transforms=None,
        testmode=False,

    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.num_classes=num_classes
        self.desired_size=image_size
        self.testmode=testmode

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path=self.images[idx]
        im = Image.open(image_path)
        im = im.resize((self.desired_size,self.desired_size), Image.NEAREST)
        if(len(np.array(im).shape)!=3):
            npim=np.array(im)
            im = np.array([npim,npim,npim])
            im=np.transpose(im,(1,2,0))
            #save_im=Image.fromarray(np.uint8(im.reshape((256,256))))
            #save_im.save('yled.png')
        im=np.array(im)
        
        im=im.clip(0,255)

        # svd and histogram matching has to be made here to ensure they choose the same picture to do svd and hist matching
        if(params.domain_adaptation_mode==1 and not self.testmode):
            
            random_style=random.choice(params.style_image_path)

            if random_style != "None":

                target_path= Path(random_style)
                target_pics=sorted(target_path.glob("*.png"))

                target_img_path = random.choice(target_pics)  #read in grayscale mode
                target_img= Image.open(target_img_path)

                randnr=random.random()

                if randnr<(1/3): #hist only
                    im=np.array(im)
                    hist_augmentation=albu.Compose([albu.HistogramMatching([str(target_img_path)], blend_ratio=(params.hist_lower_bound, params.hist_upper_bound), read_fn=readIm, p=1)])
                    im = hist_augmentation(image=im)["image"]

                else: #svd+hist
                    resized_target=target_img.resize((self.desired_size,self.desired_size), Image.NEAREST)
                    #resized_target.save('target.png')
                    u_target,s_target,vh_target=np.linalg.svd(resized_target,full_matrices=False)
                    #u_source,s_source,vh_source=np.linalg.svd(im,full_matrices=False)

                    singular_val_thresh=params.svd_threshold
                    random_thresh=random.randint(singular_val_thresh[0],singular_val_thresh[1])
                    thresholded_singular_target=s_target
                    thresholded_singular_target[0:random_thresh]=0

                    im=im+np.dot(u_target, np.dot(np.diag(thresholded_singular_target), vh_target))
                    avg=np.average(im)
                    abs_min_val=np.abs(np.min(im))
                    im=((avg*(im+abs_min_val))/(abs_min_val+avg)).clip(0,255)
                    im=np.array(im)
                    hist_augmentation=albu.Compose([albu.HistogramMatching([str(target_img_path)], blend_ratio=(params.hist_lower_bound, params.hist_upper_bound), read_fn=readIm, p=1)])
                    im = hist_augmentation(image=im)["image"]
                    save_im=Image.fromarray(np.uint8(im))
                    save_im.save('restyled.png')
        if(not self.testmode):
            im=random_darkening(im,params)

        im= np.array(im,dtype=np.uint8)

        
        result = {"image": im}
        
        if self.masks is not None:
            mask = Image.open(self.masks[idx])
            mask = mask.resize((self.desired_size,self.desired_size), Image.NEAREST)
            result["mask"]=np.array(mask)
        if(len(result["image"].shape)!=3):
            print("Wrong shape detected before transform")
        if self.transforms is not None:
            result = self.transforms(**result)
        
        if(not self.testmode and params.crop_shift_chance>random.random()):
            random_height=random.randint(65,115)
            result=crop_pad(result,0,15,random_height,256,[15,14,0,10])
        if(len(result["image"].shape)!=3):
            print("Wrong shape detected after transform")
        if self.masks is not None:
            mask_one_hot= np.transpose(np.eye(self.num_classes)[result["mask"]],(2,0,1))
            result["mask_ce"]=torch.LongTensor(result["mask"])
            result["mask"] = torch.Tensor(mask_one_hot)

        result["image"]=torch.Tensor(np.transpose(np.squeeze([result["image"]]),(2,0,1)))

        result["filename"] = image_path.name

        return result
    
# def normalize(im,testmode):
#     # dataset_name=params.dataset_name
#     # cir_mean=0.197
#     # cir_std=0.106
#     # cir_mean_train=0.182
#     # cir_std_train=0.112
#     # top_mean=0.313
#     # top_std=0.082
#     # top_mean_train=0.292
#     # top_std_train=0.088

#     # if testmode:
#     #     if dataset_name=="cirrus":
#     #         im=(im-cir_mean)/cir_std
#     #     elif dataset_name=="topcon":
#     #         im=(im-top_mean)/top_std
#     # else:
#     #     if dataset_name=="cirrus":
#     #         im=(im-cir_mean_train)/cir_std_train
#     #     elif dataset_name=="topcon":
#     #         im=(im-top_mean_train)/top_std_train
#     mean = im.mean(), 
#     #print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
# # global standardization of pixels
#     im = im - mean
#     max_val = np.max(np.abs(im))
#     im=im/max_val
#     #mean, std = im.mean(), im.std()
#     #print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
#     return im
