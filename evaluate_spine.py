# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 2/2018
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Evaluation of the spine bounding box prediction
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
## Parameters
# ------------------------------------------------------------------------------
## Generate a validation dataset
generate_validation_data = True

## Run evaluation. If False, existing file with predictions will be used
evaluate = True

## Interactive mode. If true, will display the plot inline. If false, will save
## the plot into a file
interactive = False

## Axis (coronal/sagittal)
aggregation_plane = 'coronal'

## Projection (mean/max)
aggregation_method = 'max'

## Output folder for trained model
model_snapshot = 'output/residual_ssd/cts_{}_{}_train_spine/epoch_{}.pkl'.format(
                        aggregation_plane, aggregation_method, 150)

## Use CPU only?
cpu_only = True
# ------------------------------------------------------------------------------

import os

if cpu_only:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --
import numpy as np
import cPickle
from math import sqrt
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
# --
from net import Residual_SSD
from data import OnlineSpineDataGenerator
from anchor_generator_layer import AnchorGenerator
from utils import aggregate_bboxes_ccwh
from multibox_loss import MultiboxLoss
# --

def pred2bbox(anchor, pred):
    centers = anchor[:,:,2:]*pred[:,:,:2] + anchor[:,:,:2]
    sizes = anchor[:,:,2:]*np.exp(pred[:,:,2:])
    return np.concatenate((centers, sizes), axis=-1)


if aggregation_plane == 'sagittal':
    aspect_ratios = [sqrt(2.5), sqrt(3.5)]
    scales = (5,7.5)
else:
    aspect_ratios = [sqrt(.2), sqrt(.4)]
    scales = (5,6.5)
    
## Generate tesing data
if generate_validation_data:
    aug_settings_val = {
        'use_crop': True,
        'max_crop': 0.7,
        'zmuv_mean': -103.361759224,
        'zmuv_std': 363.301491674,
        'aggregation_method': aggregation_method,
        'aggregation_plane': aggregation_plane
    }
    if aggregation_plane == 'coronal' and aggregation_method == 'mean':
        aug_settings_val['aggregation_scale'] = 0.01

    ag = AnchorGenerator(feature_stride=32,
                         offset=0,
                         aspect_ratios=aspect_ratios,
                         scale=scales)
    if aggregation_plane == 'coronal':
        min_wh_ratio=.05
    else:
        min_wh_ratio=.3

    gen_val = OnlineSpineDataGenerator(batch_size=100, imageset_name='valid_large',
                                    cts_root_path='/media/Data/Datasets/ct-spine',
                                    settings=aug_settings_val,
                                    overlap_threshold=.5,
                                    anchor_generator=ag,
                                    min_wh_ratio=min_wh_ratio)

    x_test, y_test = gen_val.Generate(shuffle=False).next()
    anchors = ag.Generate(x_test.shape)

    with open('output/validation_{}_{}.pkl'.format(aggregation_plane, aggregation_method), 'wb') as f:
        cPickle.dump((x_test,y_test,anchors), f)
# %%
if evaluate:
    ## Run predictions
    ## Load dataset
    with open('output/validation_{}_{}.pkl'.format(aggregation_plane, aggregation_method), 'rb') as f:
        (x_test, y_test, anchors) = cPickle.load(f)
    model = Residual_SSD(num_classes=2, use_bn=True, num_anchors=len(aspect_ratios)*len(scales))
    with open(model_snapshot, 'rb') as f:
        w = cPickle.load(f)
    model.set_weights(w)

    output = model.predict(x_test, batch_size=5)
    model.compile(loss=lambda y_true, y_pred: MultiboxLoss(y_true, y_pred, num_classes=2),
                  optimizer='sgd')
    loss = model.evaluate(x_test, y_test, batch_size=5)
    ## Compute results
    preds = pred2bbox(anchors,output[:,:,:4])
    gts = pred2bbox(anchors,y_test[:,:,:4])
    with open('output/predictions_{}_{}.pkl'.format(aggregation_plane, aggregation_method), 'wb') as f:
        cPickle.dump((output,loss,preds,gts), f)
#%%
## Load predictions
if not evaluate:
    with open('output/predictions_{}_{}.pkl'.format(aggregation_plane, aggregation_method), 'rb') as f:
        (output,loss,preds,gts) = cPickle.load(f)
#%%
## Compute batch IOU with various aggregation methods

## 12 different aggregation methods
# top5 max, top5 mean, top10 max, top10 mean, top15 max, top15 mean
# conf90 max, conf90 mean, conf95max, conf95mean, conf99max, conf99mean
results = np.zeros((preds.shape[0], 12))

for img_index in xrange(preds.shape[0]):
    gtbox = gts[img_index, y_test[img_index,:,5]==1]
    if gtbox.size == 0:
        print 'No GT box for image', img_index
        continue
    gtbox = gtbox[0]
    top_5 = np.argsort(-np.max(output[img_index,:,5:], axis=-1))[:5] # k = 1
    top_10 = np.argsort(-np.max(output[img_index,:,5:], axis=-1))[:10] # k = 1
    top_15 = np.argsort(-np.max(output[img_index,:,5:], axis=-1))[:15] # k = 1
    pos_90 = output[img_index,:,5] > .9
    pos_95 = output[img_index,:,5] > .95
    pos_99 = output[img_index,:,5] > .99

    aggregated = np.zeros((12,4))
    aggregated[0] = aggregate_bboxes_ccwh(preds[img_index,top_5], 'max')
    aggregated[1] = aggregate_bboxes_ccwh(preds[img_index,top_5], 'mean')
    aggregated[2] = aggregate_bboxes_ccwh(preds[img_index,top_10], 'max')
    aggregated[3] = aggregate_bboxes_ccwh(preds[img_index,top_10], 'mean')
    aggregated[4] = aggregate_bboxes_ccwh(preds[img_index,top_15], 'max')
    aggregated[5] = aggregate_bboxes_ccwh(preds[img_index,top_15], 'mean')
    if np.any(pos_90):
        aggregated[6] = aggregate_bboxes_ccwh(preds[img_index,pos_90], 'max')
        aggregated[7] = aggregate_bboxes_ccwh(preds[img_index,pos_90], 'mean')
    if np.any(pos_95):
        aggregated[8] = aggregate_bboxes_ccwh(preds[img_index,pos_95], 'max')
        aggregated[9] = aggregate_bboxes_ccwh(preds[img_index,pos_95], 'mean')
    if np.any(pos_99):
        aggregated[10] = aggregate_bboxes_ccwh(preds[img_index,pos_99], 'max')
        aggregated[11] = aggregate_bboxes_ccwh(preds[img_index,pos_99], 'mean')

    upper_left = aggregated[:,:2] - .5*aggregated[:,2:4]
    bottom_right = aggregated[:,:2] + .5*aggregated[:,2:4]
    gt_upper_left = gtbox[:2] - .5*gtbox[2:4]
    gt_bottom_right = gtbox[:2] + .5*gtbox[2:4]
    sizes = bottom_right - upper_left
    gt_size = gt_bottom_right - gt_upper_left
    areas = sizes[:,0]*sizes[:,1]
    gt_area = gt_size[0]*gt_size[1]

    over_top_left = np.maximum(gt_upper_left, upper_left)
    over_bottom_right = np.minimum(gt_bottom_right, bottom_right)
    size = np.maximum(0,over_bottom_right - over_top_left)
    inter = size[:,0] * size[:,1]
    union = gt_area + areas - inter
    iou = inter / (union+np.finfo(np.float32).eps)
    results[img_index] = iou

#%%
print 'Model loss', loss
## Plot results
labels = ['top5 - max','top5 - mean',
          'top10 - max','top10 - mean*',
          'top15 - max','top15 - mean*',
          'conf90 - max','conf90 - mean*',
          'conf95 - max','conf95 - mean',
          'conf99 - max','conf99 - mean']
plt.boxplot(results, labels=labels, vert=False)
plt.xlabel('Iou')
plt.title('Evaluation of aggregation methods')
if interactive:
    plt.show()
else:
    plt.savefig('output/aggregations_plot_{}_{}.png'.format(aggregation_plane, aggregation_method), bbox_inches='tight')
## Tabular results
for i in xrange(results.shape[1]):
    print '{:15} {:.2f} +- {:.4f} | med = {:.2f}'.format(labels[i], np.mean(results[:,i]), np.std(results[:,i]), np.median(results[:,i]))
