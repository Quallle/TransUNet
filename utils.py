import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import numpy as np
from skimage.io import imread as gif_imread
import torch
import json
from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from catalyst.contrib.nn import DiceLoss, IoULoss



class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()#.cuda()
            if torch.cuda.is_available():
                input=input.cuda()
            
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()#.cuda()
        if torch.cuda.is_available():
            input=input.cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list



class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.learning_rate = None
        self.batch_size = None
        self.num_epochs = None
        self.data_path = None
        self.img_shape = None
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def getPixelDistribution(normalized= True, verbose=True): #better implementation needed in case of more images

   giant_mask=np.array([])
   for mask in sorted(train_mask_path.glob("*.png")):
      giant_mask = np.concatenate((giant_mask,(gif_imread(mask).flatten())))
   
   giant_mask=giant_mask.flatten()

   val, count =np.unique(giant_mask, return_counts=True)
   if(normalized):
      count=count/np.sum(count)
   if(verbose):
      print("Class distribution of Pixels: ", count)
   
   return val, count

def logReweighting(pixelDistribution, device, verbose=True):
   weighting_factor=np.ones(len(pixelDistribution))
   sum_adjusted=0
   if(verbose):
      print("Class : Reweighted loss factor")
   for i,per in enumerate(pixelDistribution):
      adjusted_weight=np.max([1,int((np.log(1/per)))])
      sum_adjusted+=adjusted_weight
      weighting_factor[i]=adjusted_weight
      if(verbose):
         print(i, ': ', adjusted_weight )
   weighting_factor=torch.tensor(weighting_factor, device=device, dtype=torch.float)
   return weighting_factor

def getLogReweightedWeights(device, verbose=True):
   pixelDistribution = getPixelDistribution()
   return logReweighting(pixelDistribution, device, verbose)

def getCriterion(params):
    criterion_dict={
    "dice": DiceLoss(activation="Softmax2d"),
    "iou": IoULoss(activation="Softmax2d"),
    "ce": torch.nn.CrossEntropyLoss(),
    #"focal": FocalLossMultiClass(alpha=None, gamma=0)
    #"lovasz": LovaszLossMultiClass()
    }
    criterion = {}
    for metric in params.metrics:
        assert(metric in criterion_dict), "No matching metric found. Supported keys: 'ce','iou','dice'"
        criterion[metric]=criterion_dict[metric]
    return criterion    

def get_test_indices(params,dataset_size):
    indices=np.random.choice(np.min(params.nr_preds_to_show,dataset_size-1),params.nr_preds_to_show)
    return indices
    
def getCallbacks(params):
    
    callback_dict = {"dice": CriterionCallback(
        input_key="mask",
        prefix="loss_dice",
        criterion_key="dice"
    ), "iou":CriterionCallback(
        input_key="mask",
        prefix="loss_iou",
        criterion_key="iou"
    ),"ce": CriterionCallback(
        input_key="mask_ce",
        prefix="loss_ce",
        criterion_key="ce"
    )}
    
    callbacks=[]
    metrics={}
    assert(len(params.loss_weighting_factors)==len(params.loss_weighting_factors)), "Number of metrics and number of loss_weigthing_factors have to match."
    
    for i,metric in enumerate(params.metrics):
        assert(metric in callback_dict), "No matching metric found. Supported keys: 'ce','iou','dice'"
        callbacks+=[callback_dict[metric]]
        metrics["loss_"+ metric]=params.loss_weighting_factors[i]

    callbacks+=[DiceCallback(input_key="mask")]
    callbacks+=[MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum", # can be "sum", "weighted_sum" or "mean"
        # because we want weighted sum, we need to add scale for each loss
        metrics=metrics

    ),
    
    DiceCallback(input_key="mask"),
    IouCallback(input_key="mask")]

    return callbacks
