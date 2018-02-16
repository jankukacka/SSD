# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 2/2018
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Testing of a trained SSD model with visualization of results for spine
#  bounding box prediction in coronal projections
# ------------------------------------------------------------------------------


# --
import os
import numpy as np
import cPickle
from math import sqrt
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
# --
from net import Residual_SSD
from data import OnlineSpineDataGenerator
from anchor_generator_layer import AnchorGenerator
from utils import nms, aggregate_bboxes_ccwh
from multibox_loss import MultiboxLoss
# --

# ------------------------------------------------------------------------------
## Parameters
# ------------------------------------------------------------------------------
## Snapshot iteration number
snapshot_number = 150

## Axis (coronal/sagittal)
aggregation_plane = 'coronal'

## Projection (mean/max)
aggregation_method = 'max'

## Folder with the trained model
snapshot_folder = 'output/residual_ssd/cts_{}_{}_train_spine/'.format(aggregation_plane, aggregation_method)

## Use CPU only?
cpu_only = True

## Validation (True) or training (False) data?
use_validation = True
# ------------------------------------------------------------------------------

if cpu_only:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

aug_settings = {
    'use_crop': True,
    'max_crop': 0.7,
    'aggregation_plane': aggregation_plane,
    'aggregation_method': aggregation_method
}
if aggregation_plane == 'coronal' and aggregation_method == 'mean':
    aug_settings['aggregation_scale'] = 0.01

if use_validation:
    aug_settings['zmuv_mean'] = -103.361759224
    aug_settings['zmuv_std'] = 363.301491674
    imageset_name = 'valid_large'
else:
    aug_settings['zmuv_mean'] = 209.350884188
    aug_settings['zmuv_std'] = 353.816477769
    imageset_name = 'train_large'

if aggregation_plane == 'coronal':
    aspect_ratios = [sqrt(.2), sqrt(.4)]
    scales = (5,6.5)
    min_wh_ratio=.05
else:
    aspect_ratios = [sqrt(2.5), sqrt(3.5)]
    scales = (5,7.5)
    min_wh_ratio=.3

ag = AnchorGenerator(feature_stride=32,
                     offset=0,
                     aspect_ratios=aspect_ratios,
                     scale=scales)
gen = OnlineSpineDataGenerator(batch_size=5, imageset_name=imageset_name,
                               cts_root_path='/media/Data/Datasets/ct-spine',
                               settings=aug_settings,
                               overlap_threshold=.5,
                               anchor_generator=ag,
                               min_wh_ratio=min_wh_ratio)
x_test, y_test = gen.Generate(shuffle=False).next()

model = Residual_SSD(num_classes=2, use_bn=True, num_anchors=len(aspect_ratios)*len(scales))

with open(os.path.join(snapshot_folder, 'epoch_{}.pkl'.format(snapshot_number)), 'rb') as f:
    w = cPickle.load(f)
model.set_weights(w)

pred = model.predict(x_test, batch_size=5)
anchors = ag.Generate(x_test.shape)

#%%

def display_img_and_boxes(img, pred, gt):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    # plot GT boxes
    for i in xrange(gt.shape[0]):
        ax.add_patch(
            plt.Rectangle((gt[i][0]-.5*gt[i][2], gt[i][1]-.5*gt[i][3]),
                          gt[i][2], gt[i][3], fill=False, edgecolor='blue',
                          linewidth=2))
    for i in xrange(pred.shape[0]):
        ax.add_patch(
            plt.Rectangle((pred[i][0]-.5*pred[i][2], pred[i][1]-.5*pred[i][3]),
                          pred[i][2], pred[i][3], fill=False, edgecolor='lightgreen',
                          linewidth=1))

    plt.show()

def pred2bbox(anchor, pred):
    #result = np.zeros_like(anchor)
    centers = anchor[:,:,2:]*pred[:,:,:2] + anchor[:,:,:2]
    sizes = anchor[:,:,2:]*np.exp(pred[:,:,2:])
    return np.concatenate((centers, sizes), axis=-1)

img_index = 0
img = x_test[img_index,:,:,0]
preds = pred2bbox(anchors,pred[:,:,:4])[img_index]
gts = pred2bbox(anchors,y_test[:,:,:4])[img_index]
top_k = np.argsort(-np.max(pred[img_index,:,5:], axis=-1))[:5] # k = 10
pos = pred[img_index,:,5] > .90

aggregated = aggregate_bboxes_ccwh(preds[pos,:4], 'mean')
display_img_and_boxes(img, aggregated[np.newaxis,:], gts[y_test[img_index,:,5]==1])
aggregated = aggregate_bboxes_ccwh(preds[top_k,:4], 'mean')
display_img_and_boxes(img, aggregated[np.newaxis,:], gts[y_test[img_index,:,5]==1])


#%%
keep = nms(pred[:,:,[0,1,2,3,5]][img_index, pos], sigma=0.01, cutoff=0.5)
keep = np.where(pos)[0][keep]
plt.hist(pred[img_index,:,5])
plt.show()
# print pred[img_index, top_k]
# print y_test[img_index,:,:4]
display_img_and_boxes(img, preds[top_k], gts[y_test[img_index,:,5]==1])
display_img_and_boxes(img, preds[keep], gts[y_test[img_index,:,5]==1])

#%%
