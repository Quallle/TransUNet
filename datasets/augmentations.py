import albumentations as albu
import os
from albumentations.augmentations.transforms import ElasticTransform, ISONoise
import cv2
import random
import numpy as np

example_saved_count=0


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def get_augmentations(params):
    image_size=params.img_shape
    augmentations=albu.Compose([albu.OneOf([albu.RandomSizedCrop(min_max_height = (image_size  - 40, image_size  - 40),height = image_size,width = image_size,p = 0.1),
            albu.PadIfNeeded(min_height = image_size, min_width = image_size, p = 1)], p = 1),

        
        albu.VerticalFlip(p = 0.2),
        albu.RandomRotate90(p = 0.2),
        albu.Rotate(p = 0.5),
        albu.RandomBrightnessContrast(brightness_limit = 0.6, contrast_limit = 0.6, p = 0.1),
        albu.GaussNoise(p = 0.1, var_limit = (10.0, 25.0)),
        albu.ISONoise(color_shift = (0.01, 0.5), intensity = (0.1, 0.9), p = 0.1),
        albu.RandomGamma(gamma_limit = (50, 150), p = 0.1),
        albu.GridDistortion(p=0.2),
        albu.ElasticTransform(p=0.2)])

    if(params.train_transforms==1):
        train_transforms = albu.Compose([augmentations])
    else:
        train_transforms = None

    if(params.valid_transforms==1):
        validation_transforms = albu.Compose([augmentations])
    else:
        validation_transforms = None

    if(params.test_transforms==1):
        test_transforms = albu.Compose([augmentations])
    else:
        test_transforms = None

    
    
    return train_transforms, validation_transforms, test_transforms

def readIm(imagepath):
  image = cv2.imread(str(imagepath),1)
  return image

def random_darkening(im,params):
    if(random.random()<params.random_darkening_prob):
        gradient=create_random_gradient_image(len(im[0]),params.min_darkness_width,params.max_darkness_width)
        return add_gradient_to_image(im,gradient)
    else:
        return im

def add_gradient_to_image(im,gradient):

    rand_int = random.randint(0,1)
    multiplying_factor = np.ones_like(im).astype(float)
   
    if rand_int==0:
        gradient=np.flip(gradient)
        multiplying_factor[:,0:len(gradient)]=gradient
    else:
        index=len(im)-len(gradient)
        multiplying_factor[:,index:]=gradient

    im=np.multiply(im,multiplying_factor).astype(np.uint8)
    return im


def create_random_gradient_image(im_height,min_width,max_width):
    rand_int=random.randint(min_width,max_width)
    x = np.linspace(1, 0.4, rand_int)
    #gradient = np.tile(x, (im_height, 1))
    return x



def crop_pad(result,image_pad_label,mask_pad_label,crop_height,crop_width,ignore_values=None):
    
    size=result["image"].shape

    cropping=albu.Compose([(albu.CropNonEmptyMaskIfExists(crop_height,crop_width,ignore_values=ignore_values))])
    result = cropping(**result)

    position=(random.randint(0,1),0.5)

    result["image"]= pad_position(result["image"],image_pad_label,size=size,position=position)
    result['mask']=pad_position(result["mask"],mask_pad_label,size=size,position=position)

    return result


def pad_position(image,pad_label,size,position=(0.5,0.5)):
    eps=0.00001
    divisors= (1/(position[0]+eps), 1/(position[1]+eps))
    padded_im=pad(image,size[0],size[1],divisors,value=pad_label)
    return padded_im
    

def pad(img, min_height, min_width,divisors=(2.,2.),border_mode=cv2.BORDER_CONSTANT,value=None):
    height, width = img.shape[:2]

    if height < min_height:
        h_pad_top = int((min_height - height) / divisors[0])
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / divisors[1])
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = pad_with_params(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value)

    if img.shape[:2] != (max(min_height, height), max(min_width, width)):
        raise RuntimeError(
            "Invalid result shape. Got: {}. Expected: {}".format(
                img.shape[:2], (max(min_height, height), max(min_width, width))
            )
        )

    return img

def pad_with_params(
    img,
    h_pad_top,
    h_pad_bottom,
    w_pad_left,
    w_pad_right,
    border_mode=cv2.BORDER_CONSTANT,
    value=None,
):
    img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value=value)
    return img
