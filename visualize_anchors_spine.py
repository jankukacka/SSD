import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from math import sqrt
from anchor_generator_layer import AnchorGenerator
from anchor_assignment import Match#, _iou, _assign_boxes
from data import OnlineSpineDataGenerator

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

aug_settings_train = {
    'use_crop': True,
    'crop_max': 0.7,
    'zmuv_mean': 209.350884188,
    'zmuv_std': 353.816477769
}
ag = AnchorGenerator(feature_stride=32, offset=0, aspect_ratios=[sqrt(2.5), sqrt(3.5)], scale=(5,7.5))
data_gen = OnlineSpineDataGenerator(batch_size=1, imageset_name='train_large',
                                cts_root_path='/media/Data/Datasets/ct-spine',
                                settings=aug_settings_train, return_anchors=True,
                                anchor_generator=ag,
                                overlap_threshold=.5)
data = next(data_gen.Generate())

shape = data[0][0].shape
#anchors = ag.Generate((1, shape[0], shape[1], shape[2]))
offsets = data[1][:,:,:-4]
anchors = data[1][:,:,-4:]
# bboxes = np.expand_dims(data[1][0], 0)
# offsets = Match(bboxes, anchors, 2, .3, (shape[1], shape[0]))

# print offsets[offsets[:,:,5] == 1, :4]

def display_img_and_boxes(img, pred, gt):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    # plot GT boxes
    if gt is not None:
        for i in xrange(gt.shape[0]):
            ax.add_patch(
                plt.Rectangle((gt[i][0]-.5*gt[i][2], gt[i][1]-.5*gt[i][3]),
                              gt[i][2], gt[i][3], fill=False, edgecolor='lightblue',
                              linewidth=1))
    for i in xrange(pred.shape[0]):
        ax.add_patch(
            plt.Rectangle((pred[i][0]-.5*pred[i][2], pred[i][1]-.5*pred[i][3]),
                          pred[i][2], pred[i][3], fill=False, edgecolor='red',
                          linewidth=1))

    plt.show()

def pred2bbox(anchor, pred):
    #result = np.zeros_like(anchor)
    centers = anchor[:,:,2:]*pred[:,:,:2] + anchor[:,:,:2]
    sizes = anchor[:,:,2:]*np.exp(pred[:,:,2:])
    return np.concatenate((centers, sizes), axis=-1)

img = data[0][0,:,:,0]

single_image = pred2bbox(anchors, offsets[:,:,:4])[0]
top = offsets[0,:,5]==1
bottom = offsets[0,:,4]==1
print 'pos_boxes:', np.sum(top), 'neg_boxes:', np.sum(bottom)
if np.sum(top) > 0:
    print 'target aspect:', single_image[top][0,2]/single_image[top][0,3]

negatives = single_image[bottom]
#negatives = negatives[np.random.choice(negatives.shape[0], 3*np.sum(top), replace=False)]
display_img_and_boxes(img, single_image[top], data[1][0])
# display_img_and_boxes(img, anchors[0,::43], None)
# display_img_and_boxes(img, negatives, None)
display_img_and_boxes(img, anchors[0][top], data[1][0])
# print np.any(anchors[0][top][:,:2] - .5*anchors[0][top][:,2:] < 0, axis=-1, keepdims=True)
#%%
aug_settings_train = {
    'use_crop': True,
    'zmuv_mean': 209.350884188,
    'zmuv_std': 353.816477769,
    'crop_max': 0.7,
    'aggregation_plane': 'coronal'
}
ag = AnchorGenerator(feature_stride=32, offset=0, aspect_ratios=[sqrt(0.4), sqrt(0.25)], scale=(5,7.5))
data_gen = OnlineSpineDataGenerator(batch_size=1, imageset_name='train_large',
                                cts_root_path='/media/Data/Datasets/ct-spine',
                                settings=aug_settings_train,
                                anchor_generator=ag,
                                overlap_threshold=.6,
                                match_anchors=False, min_wh_ratio=.05)
aspects = []
scales = []
for _ in xrange(100):
    data = next(data_gen.Generate())
    box = data[1][0,0,:4]
    aspect = box[2]/box[3]
    scale = box[2]/(32*sqrt(aspect))
    aspects.append(aspect)
    scales.append(scale)
    print aspect, scale
plt.hist(aspects)
plt.show()
plt.hist(scales)
plt.show()
