import random
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec
import numpy as np
from PIL import Image
import os
from sklearn.metrics import confusion_matrix
from catalyst.dl import Callback, CallbackOrder, IRunner
import torch

from pathlib import Path
from typing import Dict, Any
from os import PathLike
import sys



def mp(entry,mapper_dict):
    return mapper_dict[entry] if entry in mapper_dict else entry
mp=np.vectorize(mp)

replacements_dict={3:1, 4:2,7:3,8:3,11:3, 1:7, 2:4,5:5,6:6,9:9,10:10,12:12,13:13,14:14,15:15} #for visualization purposes
replacements_dict_labels={3:1, 4:2,7:3,8:3,11:3, 1:0, 2:0,5:0,6:0,9:0,10:0,12:0,13:0,14:0,15:0}
def show_preds(images, predictions, index, visualization_indices, remap=False, save_path=None):
    names=images['filename']
    if 'mask_ce' in images:
        gtmasks=images['mask_ce'].cpu().numpy()
        n_images=zip(images['image'],gtmasks)
    else:
        n_images=images['image']
    
    
    predictions=predictions['softmax']
    for i,(image, prediction,name) in enumerate(zip(n_images, predictions,names)):
        if (index+i in visualization_indices):
            mask = np.argmax(prediction,axis=0)
            if(remap):
                mask=mp(mask,replacements_dict)
            if(len(image)>1):
                gt_mask=np.squeeze(image[1])
            else:
                gt_mask= None
            image= np.squeeze(image[0])
            plot_image_label_prediction([image,gt_mask,mask],model_dir=save_path,mode="test",filename=name,remap=remap)


def getLabelsForMatrix(aggregatedgt,predictions, params):
    if(params.remap==1):   
        aggregatedgt=mp(aggregatedgt,replacements_dict)
        predictions=mp(predictions,replacements_dict)
    labels=[]
    for l in np.unique([[aggregatedgt.flatten()],[predictions.flatten()]]):
        labels.append(str(int(l)))
    return labels

def plot_iou(iou_list,save_path):
    plt.figure()
    bars=plt.bar(np.arange(0,len(iou_list)),iou_list.values(),tick_label=list(iou_list.keys()))
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, np.round(yval,decimals=3))
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+'/iou.png',bbox_inches='tight')
        print("Saved IOU statistics at ", save_path,'/iou.png')
    plt.close()



def plot_image_label_prediction(records, model_dir, mode, filename, remap=False):
    """
    :param records: list containing numpy array of image, label and prediction
    :param model_dir: directory of model where to save images
    :param mode: str: train or test
    :param filename: str: filename of image
    :return: save images in directory for inspection
    """

    # set prediction to black if not given
    if len(records) < 3:
        records.append(np.zeros(records[0].shape))

    seg_cmap, seg_norm, bounds = color_mappings(remap)
    #print(type(records[1]))
    if records[1] is not None:
        fig = plt.figure(figsize=(16, 4))

        gs = gridspec.GridSpec(nrows=1,
                            ncols=3,
                            figure=fig,
                            width_ratios=[1, 1, 1],
                            height_ratios=[1],
                            wspace=0.3,
                            hspace=0.3)

        # turn image to 3 channel
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(records[0][0],cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("oct")

        # check label shape
        
        if len(records[1].shape) == 3:
            records[1] = records[1][:, :, 0]

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(records[1], cmap=seg_cmap, norm=seg_norm,interpolation='nearest')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("ground truth")

        ax3 = fig.add_subplot(gs[0, 2])
        
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("prediction")
        colorbar_im=ax3.imshow(records[2], cmap=seg_cmap, norm=seg_norm,interpolation='nearest')

        # set colorbar ticks
        tick_loc_array = np.arange(len(bounds)) + 0.5
        tick_loc_list = tick_loc_array.tolist()
        tick_list = np.arange(len(bounds)).tolist()
        c_bar = plt.colorbar(colorbar_im, cmap=seg_cmap, norm=seg_norm, boundaries=bounds)

        #set ticks
        c_bar.set_ticks(tick_loc_list)
        c_bar.ax.set_yticklabels(tick_list)

        if not os.path.exists(os.path.join(model_dir, mode + "_records")):
            os.makedirs(os.path.join(model_dir, mode + "_records"))
    if not os.path.exists(os.path.join(model_dir, mode + "_raw")):
        os.makedirs(os.path.join(model_dir, mode + "_raw"))
    if not os.path.exists(os.path.join(model_dir,"numpy_preds")):
        print(os.path.join(model_dir,"numpy_preds"))
        os.makedirs(os.path.join(model_dir,"numpy_preds"))
    if records[1] is not None:
        plt.savefig(os.path.join(model_dir, mode + "_records", filename),bbox_inches = 'tight' , pad_inches = 0)
    plt.imsave(os.path.join(model_dir, mode + "_raw", filename), records[2],cmap=seg_cmap,vmin=0,vmax=16)
    plt.close()
    
# def show_random(images: List[Path], masks: List[Path], transforms=None) -> None:
#     length = len(images)
#     index = random.randint(0, length - 1)
#     show(index, images, masks, transforms)


def show_fda(im_src, imtarget, restyled_src, beta,name,save_path=None):
    plt.figure(figsize=(60, 50), constrained_layout=False)
    plt.subplot(131), plt.imshow(im_src.astype(np.uint8)), plt.title("Source Image")
    plt.subplot(132), plt.imshow(imtarget.astype(np.uint8)), plt.title("Target Style Image")
    plt.subplot(133), plt.imshow(restyled_src, "gray"), plt.title("Source Content Target Style beta=" + str(beta))
    if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path+"/"+"beta=" + str(beta)+name, bbox_inches='tight')
    plt.close()

def plotConfusionMatrix(cm, target_names, params, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(40, 30))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(params.logdir+"/"+"ConfusionMatrix.png",bbox_inches='tight')
    plt.show()


def showConfusionMatrix(test_dataset, predictions, params):
    matrix, labels = getConfusionMatrix(test_dataset, predictions, params)
    plotConfusionMatrix(matrix, labels, params)
    


class CustomInferCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Internal)
        self.heatmap = None
        self.counter = 0

    def on_loader_start(self, runner: IRunner):
        self.predictions = None
        self.counter = 0

    def on_batch_end(self, runner: IRunner):
        # data from the Dataloader
        # image, mask = runner.input["image"], runner.input["mask"]
        #logits is a misleading name as it is raw output, not logits. changing the name leads to a bug in catalyst.
        output = runner.output["logits"] 
        probabilities = torch.sigmoid(output)

        self.heatmap = (
            probabilities 
            if self.heatmap is None 
            else self.heatmap + probabilities
        )
        self.counter += len(probabilities)

    def on_loader_end(self, runner: IRunner):
        self.heatmap = self.heatmap.sum(axis=0)
        self.heatmap /= self.counter


def getConfusionMatrix(dataset, predictions,params, sample_size=1):
    aggregatedgt=np.array([])
    aggregatedpred=np.array([])
    for i,(features, output) in enumerate(zip(dataset, predictions)):
        if(random.random()<sample_size):
            gtmask = np.array(features['mask_ce'])
            aggregatedgt = np.append(aggregatedgt,gtmask.flatten())
            mask_ = np.array(torch.argmax(torch.from_numpy(output).sigmoid(),dim=0))
            aggregatedpred=np.append(aggregatedpred,mask_.flatten())
    if(params.remap==1):   
        aggregatedgt=mp(aggregatedgt,replacements_dict)
        predictions=mp(predictions,replacements_dict)

    confusion=confusion_matrix(aggregatedgt.flatten(),aggregatedpred.flatten())
    labels=getLabelsForMatrix(aggregatedgt,aggregatedpred, params)
    return confusion, labels




def color_mappings(remap=False):
    #if(remap): return color_mappings_small()
    color_palett = np.array([[148., 158., 167.],
                             [11., 151., 199.],
                             [30., 122., 57.],
                             [135., 191., 234.],
                             [37., 111., 182.],
                             [156., 99., 84.],
                             [226., 148., 60.],
                             [203., 54., 68.],
                             [192., 194., 149.],
                             [105., 194., 185.],
                             [205., 205., 205.],
                             [140., 204., 177.],  # Serous PED
                             [183., 186., 219.],  # other artifact
                             [114, 137, 218],  # fibrosis
                             [209., 227., 239.],
                             [226., 233., 48.]])

    color_palett_norm = color_palett / 255  # (np.max(color_palett)-np.min(color_palett))
    custom_cmap = colors.ListedColormap(
        color_palett_norm
    )

    # set counts and norm
    array_bounds = np.arange(color_palett.shape[0] + 1) - 0.1
    bounds = array_bounds.tolist()
    norm = colors.BoundaryNorm(bounds, custom_cmap.N)
    return custom_cmap, norm, bounds

def write_all_eval_files(params):

    current_vol=None
    prev_vol=None
    oct_volume=None

    all_predictions = sorted(Path(params.logdir+"_retouch").glob("*.png"))
    for i,im_path in enumerate(all_predictions):

        current_vol=str(im_path).split("\\")[-1].split("/")[-1].split("-")[0]
        im=Image.open(im_path)

        if (current_vol!=prev_vol and oct_volume != None):
            if len(str(prev_vol))==1:
                prev_vol="0"+str(prev_vol)
            write_mhd_file(params.logdir+"_retouch/TEST0"+str(prev_vol)+".mhd",np.array(oct_volume))
            

        if(current_vol!=prev_vol):
            oct_volume=[np.array(im,dtype=np.uint8)]

        else:
            oct_volume.append(np.array(im,dtype=np.uint8))

        if i==(len(all_predictions)-1):
            if len(str(prev_vol))==1:
                prev_vol="0"+str(prev_vol)
            write_mhd_file(params.logdir+"_retouch/TEST0"+str(prev_vol)+".mhd",np.array(oct_volume))


        prev_vol=current_vol
        os.unlink(im_path)

#meta-file writer taken from https://github.com/yanlend/mhd_utils
def write_meta_header(filename: PathLike, meta_dict: Dict[str, Any]):
    """
    Write the MHD meta header file
    :param filename: file to write
    :param meta_dict: dictionary of meta data in MetaImage format
    """
    header = ''
    # do not use tags = meta_dict.keys() because the order of tags matters
    tags = ['ObjectType', 'NDims', 'BinaryData',
            'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
            'TransformMatrix', 'Offset', 'CenterOfRotation',
            'AnatomicalOrientation', 'ElementSpacing',
            'DimSize', 'ElementNumberOfChannels', 'ElementType', 'ElementDataFile',
            'Comment', 'SeriesDescription', 'AcquisitionDate',
            'AcquisitionTime', 'StudyDate', 'StudyTime']
    for tag in tags:
        if tag in meta_dict.keys():
            header += '%s = %s\n' % (tag, meta_dict[tag])
    with open(filename, 'w') as f:
        f.write(header)


def write_mhd_file(filename: PathLike, data: np.ndarray, **meta_dict):
    """
    Write a meta file and the raw file.
    The byte order of the raw file will always be in the byte order of the system. 
    :param filename: file to write
    :param meta_dict: dictionary of meta data in MetaImage format
    """
    assert filename[-4:] == '.mhd' 
    meta_dict['ObjectType'] = 'Image'
    meta_dict['BinaryData'] = 'True'
    meta_dict['BinaryDataByteOrderMSB'] = 'False' if sys.byteorder == 'little' else 'True'
    if data.dtype == np.float32:
        meta_dict['ElementType'] = 'MET_FLOAT'
    elif data.dtype == np.double or data.dtype == np.float64:
        meta_dict['ElementType'] = 'MET_DOUBLE'
    elif data.dtype == np.byte:
        meta_dict['ElementType'] = 'MET_CHAR'
    elif data.dtype == np.uint8 or data.dtype == np.ubyte:
        meta_dict['ElementType'] = 'MET_UCHAR'
    elif data.dtype == np.short or data.dtype == np.int16:
        meta_dict['ElementType'] = 'MET_SHORT'
    elif data.dtype == np.ushort or data.dtype == np.uint16:
        meta_dict['ElementType'] = 'MET_USHORT'
    elif data.dtype == np.int32:
        meta_dict['ElementType'] = 'MET_INT'
    elif data.dtype == np.uint32:
        meta_dict['ElementType'] = 'MET_UINT'
    else:
        raise NotImplementedError("ElementType " + str(data.dtype) + " not implemented.")
    dsize = list(data.shape)
    if 'ElementNumberOfChannels' in meta_dict.keys():
        element_channels = int(meta_dict['ElementNumberOfChannels'])
        assert(dsize[-1] == element_channels)
        dsize = dsize[:-1]
    else:
        element_channels = 1
    dsize.reverse()
    meta_dict['NDims'] = str(len(dsize))
    meta_dict['DimSize'] = dsize
    meta_dict['ElementDataFile'] = str(Path(filename).name).replace('.mhd', '.raw')
    print(str(Path(filename).name).replace('.mhd', '.raw'))

    # Tags that need conversion of list to string
    tags = ['ElementSpacing', 'Offset', 'DimSize', 'CenterOfRotation', 'TransformMatrix']
    for tag in tags:
        if tag in meta_dict.keys() and not isinstance(meta_dict[tag], str):
            meta_dict[tag] = ' '.join([str(i) for i in meta_dict[tag]])
    write_meta_header(filename, meta_dict)

    # Compute absolute path to write to
    pwd = Path(filename).parents[0].resolve()
    data_file = Path(meta_dict['ElementDataFile'])
    if not data_file.is_absolute():
        data_file = pwd / data_file

    # Dump raw data
    data = data.reshape(dsize[0], -1, element_channels)
    with open(data_file, 'wb') as f:
        data.tofile(f)

def save_numpy_prediction(images,predictions,params):
    names=images['filename']
    predictions=predictions['softmax']
    if not os.path.exists(params.logdir+"/predictions/"+"numpy_preds/"):
        os.makedirs(params.logdir+"/predictions/"+"numpy_preds/")
    for prediction,name in zip (predictions,names):
        mask = np.array(np.argmax(prediction,axis=0),dtype=np.uint8)
        np.save(params.logdir+"/predictions/"+"numpy_preds/"+name,mask)
        

def save_raw_im(images,predictions,params):
    if not os.path.exists(params.logdir+"_retouch/"):
            os.makedirs(params.logdir+"_retouch/")

    names=images['filename']
    predictions=predictions['softmax']

    for prediction,name in zip (predictions,names):
        mask = np.array(np.argmax(prediction,axis=0),dtype=np.uint8)
        mask=mp(mask,replacements_dict_labels).astype(np.uint8)
        # print(mask.dtype)
        # print(np.unique(mask))
        # print(mask.shape)
        # print(name)
        vol=name.split("-")[0]
        im_nr=name.split("-")[1].replace(".png","")
        if(len(im_nr)==1):
            im_nr="00"+im_nr
        elif(len(im_nr)==2):
            im_nr="0"+im_nr
        mask=Image.fromarray(mask)
        mask=mask.resize(get_reference_size(name,params.test_image_path),Image.NEAREST)
        mask.save(params.logdir+"_retouch/"+vol+"-"+im_nr+".png")

    

#resize to original size after predicting for retouch challenge evaluation
def get_reference_size(im_path,ref_path):
    im_name=str(im_path).split("\\")[-1]
    im=Image.open(ref_path+"/"+im_name)
    #print(im.size)
    
    return im.size
