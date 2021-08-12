import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda import is_available
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import SegmentationDataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from numpy.core.numeric import zeros_like
from torch.nn import Softmax2d
import copy
from plotting_utils import save_numpy_prediction, show_preds,save_raw_im
from utils import Params
import csv
from sklearn.metrics import f1_score,jaccard_score
from pathlib import Path
from catalyst import utils
from catalyst.dl import SupervisedRunner


params=Params("./params.json")


replacements_dict={3:1, 4:2,7:3,8:3,11:3, 1:0, 2:0,5:0,6:0,9:0,10:0,12:0,13:0,14:0,15:0}

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"

def get_split(filepath,im_array,mask_array):
    new_mask_arr=[]
    new_im_arr=[]
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for im in im_array:
                if(row[1] in str(im)):
                    new_im_arr.append(im)
                    break
            for mask in mask_array:
                if(row[1] in str(mask)):
                    new_mask_arr.append(mask)
                    break
    return new_im_arr,new_mask_arr
    
def predict_and_visualize(dataloader,model,runner,params,dataset_size,evalmode=False):
    visualization_indices = np.random.choice(dataset_size, params.nr_preds_to_show)
    ensemble_prediction(dataloader,model, runner, params,visualization_indices,evalmode)

def ensemble_prediction(dataloader, model, runner, params,visualization_indices, evalmode=False):
    if len(dataloader.dataset.masks)>0:
        masks_loaded=True
    else:
        masks_loaded=False
    print("Starting prediction..")

    softmax=Softmax2d()

    #our special case when we remap to only 4 classes
    if(params.remap):
        nr_classes=4
    else:
        nr_classes=params.num_classes

    predictions=np.zeros((params.batch_size,params.num_classes,params.img_shape,params.img_shape))
    predictors=np.array([])
    pred_dict={}

    #following code could be used for ensemble prediction of multiple models
    if(evalmode):

        predictors=np.append(predictors,copy.deepcopy(runner).predict_loader(loader=dataloader, model=copy.deepcopy(model)))

    else:
        predictors=[runner.predict_loader(loader=dataloader, model=model)]

    nr_predictors=len(predictors)
    
    for i,all_data in enumerate(zip(dataloader,*predictors)):
        ensemble_pred=dict()
        preds=all_data[1:]

        #logits is a misleading name as it is raw output, not logits. changing the name leads to a bug in catalyst.
        ensemble_pred['softmax']=zeros_like(preds[0]['logits'].cpu().numpy())
        image_data=all_data[0]

        for single_batch in preds:
            single_batch['softmax']=softmax(single_batch['logits']).cpu().numpy() 
        for single_batch in preds:
            ensemble_pred['softmax']= ensemble_pred['softmax']+(single_batch['softmax']/nr_predictors)

        if(i==0):
            if(masks_loaded):
                dice_score,iou_score,score_list=dice(image_data,ensemble_pred,params)
            class_occurence_dict=occurence_checker(image_data,ensemble_pred,params)

        elif masks_loaded:
            dice_now,iou_now,score_list_now=dice(image_data,ensemble_pred,params)
            dice_score+=dice_now
            iou_score+=iou_now
            score_list+=score_list_now

        if(i>0):
            class_occurence_dict.update(occurence_checker(image_data,ensemble_pred,params))

        if masks_loaded:
            pred_dict.update(get_pixel_predictions(image_data, ensemble_pred, params))

        show_preds(image_data,ensemble_pred,i,visualization_indices,save_path=params.logdir+"/predictions")

        if(params.save_raw):
            save_raw_im(image_data,ensemble_pred,params)
        if(params.save_numpy):
            save_numpy_prediction(image_data,ensemble_pred,params)
        
        if(params.remap==1):
            show_preds(image_data,ensemble_pred,i,visualization_indices,remap=True,save_path=params.logdir+"/predictionsRemapped")
        print("\r {}".format("Predicting: Batch " +str(i+1) + " of " + str(int(len(dataloader.dataset.images)/params.batch_size))+ " completed."), end="")
      
    if(masks_loaded):
        iou_score=absToPercent(iou_score,nr_classes)
        dice_score=absToPercent(dice_score,nr_classes)
        score_list.sort(key=get_mean)
        print("IOU stats: ", iou_score)
        print("Dice stats: ", dice_score)
    #print("Score list: ", score_list)
    

        with open('score_file.csv', mode='w') as score_file:
            writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for score in score_list:
                writer.writerow(score)
        with open('pixel_pred_file.csv', mode='w') as score_file:
            writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for key in pred_dict:
                writer.writerow((key,pred_dict[key]))

    with open('occurence_file.csv', mode='w') as score_file:
        writer = csv.writer(score_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for key in class_occurence_dict:
            writer.writerow((key,class_occurence_dict[key]))
            
    return predictions

def absToPercent(dice_scores,nr_classes):
    mean_score=dict()
    mean_list=[]
    for i in range(nr_classes):
        cleaned_list=[image_score[i] for image_score in dice_scores if image_score[i]<1.0]
        if(len(cleaned_list)>0):
            mean_score[i]=np.mean(cleaned_list)
        else:
            mean_score[i]=None
    
    for i in mean_score.keys():
        if(i!=0 and mean_score[i]!=None):#0 is background
            mean_list.append(mean_score[i])
    mean_score['mean']=np.mean(mean_list) 
    return mean_score

def mp(entry,mapper_dict):
    return mapper_dict[entry] if entry in mapper_dict else entry
mp=np.vectorize(mp)

def occurence_checker(images, predictions,params):
    filenames=images['filename']
    im_dict={}
    for (values, name) in zip(predictions['softmax'],filenames):
        mask = np.array(torch.argmax(torch.from_numpy(values),dim=0))

        arr=np.zeros(params.num_classes)
        
        for i in range(params.num_classes):
            if(i in mask):arr[i]=1
        im_dict[name]=arr
    return im_dict

def dice(images, predictions, params):
    if('mask_ce' not in images):
        return None,None,None
    filenames=images['filename']
    images=images['mask_ce']
    predictions=predictions['softmax']
    #dice_list = np.zeros((params.num_classes,1))
    dice_list=[]
    iou_list=[]
    score_list=[]
    nr_classes=params.num_classes

    if(params.remap==1):
        nr_classes=4
    else:
        nr_classes=params.num_classes

    for (image, values, name) in zip(images, predictions,filenames):   

            gtmask=np.array(image).flatten()
            mask = np.array(torch.argmax(torch.from_numpy(values),dim=0)).flatten()

            if(params.remap==1):
                mask=mp(mask,replacements_dict)

            dice_score_image=f1_score(gtmask,mask,labels=range(nr_classes),average=None,zero_division=1)
            iou_score_image=jaccard_score(gtmask,mask,labels=range(nr_classes),average=None,zero_division=1)

            dice_list.append(dice_score_image)
            iou_list.append(iou_score_image)
            dice_score_image=np.append(dice_score_image,np.mean(dice_score_image[dice_score_image<1][1:]))
            score_list.append([name,dice_score_image])

    return dice_list, iou_list, score_list

def get_mean(score):
    return score[1][len(score[1])-1]

def get_pixel_predictions(images, predictions, params):

    
    pred_dict={}

    if('mask_ce' not in images):
        return None,None,None
    filenames=images['filename']
    images=images['mask_ce']
    predictions=predictions['softmax']

    
    nr_classes=params.num_classes
    

    for (image, values, name) in zip(images, predictions,filenames):
        pixel_list = np.zeros((params.num_classes,2))
        if(params.remap==1):
            pixel_list = np.zeros((4,2))
            nr_classes=4
        
        for sem_class in range(nr_classes):

            gtmask=np.array(image).flatten()
            mask = np.array(torch.argmax(torch.from_numpy(values),dim=0)).flatten()

            if(params.remap==1):
                mask=mp(mask,replacements_dict)

            pred_inds = (mask == sem_class)
            target_inds = (gtmask == sem_class)

            intersection_now = (pred_inds[target_inds]).sum()
            union_now = pred_inds.sum().item() + target_inds.sum().item()
            pixel_list[sem_class]=np.array([intersection_now,union_now])

        pred_dict[name]=pixel_list

    return pred_dict

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': SegmentationDataset,
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': params.num_classes,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)#.cuda()
    if torch.cuda.is_available():
        net=net.cuda()

    snapshot=params.model_path
    if(torch.cuda.is_available()):
        net.load_state_dict(torch.load(snapshot))
    else:
        net.load_state_dict(torch.load(snapshot,map_location=torch.device("cpu")))

    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    images = sorted(Path(params.train_image_path).glob("*.png"))
    masks = sorted(Path(params.train_mask_path).glob("*.png"))
    train_path=params.train_id_path
    test_path=params.test_id_path
    train_images,train_masks=get_split(train_path,images,masks)
    print("Generated trainingset...")
    print("Generated validationset...")
    test_images,test_masks=get_split(test_path,images,masks)

    db_test = SegmentationDataset(
      images = test_images,
      masks = test_masks,
      transforms = None,
      num_classes = params.num_classes,
      image_size=params.img_shape,
      testmode=True
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    device = utils.get_device()
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")
    predict_and_visualize(testloader,net,runner,params,len(db_test),evalmode=True)




