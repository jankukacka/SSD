# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 2/2018
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Testing of a trained SSD model with visualization of results for spine
#  bounding box prediction
# ------------------------------------------------------------------------------


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
from utils import nms, aggregate_bboxes_ccwh
from multibox_loss import MultiboxLoss
# --

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


aug_settings_train = {
    'use_crop': True,
    'zmuv_mean': 209.350884188,
    'zmuv_std': 353.816477769
}
aug_settings_val = {
    'use_crop': False,
    'zmuv_mean': -103.361759224,
    'zmuv_std': 363.301491674,
    'aggregation_plane': 'coronal'
}
aspect_ratios = [sqrt(2.5), sqrt(3.5)]
scales = (5,7.5)
ag = AnchorGenerator(feature_stride=32,
                     offset=0,
                     aspect_ratios=aspect_ratios,
                     scale=scales)
gen_train = OnlineSpineDataGenerator(batch_size=2, imageset_name='train_large',
                                cts_root_path='/media/Data/Datasets/ct-spine',
                                settings=aug_settings_train,
                                overlap_threshold=.4,
                                anchor_generator=ag)
gen_val = OnlineSpineDataGenerator(batch_size=5, imageset_name='valid_large',
                                cts_root_path='/media/Data/Datasets/ct-spine',
                                settings=aug_settings_val,
                                overlap_threshold=.4,
                                anchor_generator=ag)

x_test, y_test = gen_val.Generate(shuffle=False).next()

model = Residual_SSD(num_classes=2, use_bn=True, num_anchors=len(aspect_ratios)*len(scales))

with open('output/residual_ssd/cts_sagittal_train_spine/epoch_100.pkl', 'rb') as f:
    w = cPickle.load(f)
model.set_weights(w)


pred = model.predict(x_test, batch_size=5)
print np.sum(pred[:,:,5:]>.9, axis=(1,2))

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

img_index = 4
img = x_test[img_index,:,:,0]
preds = pred2bbox(anchors,pred[:,:,:4])[img_index]
gts = pred2bbox(anchors,y_test[:,:,:4])[img_index]
top_k = np.argsort(-np.max(pred[img_index,:,5:], axis=-1))[:5] # k = 1
pos = pred[img_index,:,5] > .9

aggregated = aggregate_bboxes_ccwh(preds[pos,:4], 'mean')
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
# --
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
# --
from data import OnlineSpineDataGenerator
from anchor_generator_layer import AnchorGenerator
from utils import aggregate_bboxes_ccwh

aug_settings_val = {
    'use_crop': False,
    'zmuv_mean': -103.361759224,
    'zmuv_std': 363.301491674,
    'aggregation_plane': 'coronal',
    'aggregation_scale':0.01
}
aspect_ratios = [sqrt(2.5), sqrt(3.5)]
scales = (5,7.5)
ag = AnchorGenerator(feature_stride=32,
                     offset=0,
                     aspect_ratios=aspect_ratios,
                     scale=scales)
gen_val = OnlineSpineDataGenerator(batch_size=1, imageset_name='valid_large',
                                cts_root_path='/media/Data/Datasets/ct-spine',
                                settings=aug_settings_val,
                                overlap_threshold=.01,
                                anchor_generator=ag)

x_test, y_test = gen_val.Generate(shuffle=False).next()
anchors = ag.Generate(x_test.shape)

gts = pred2bbox(anchors,y_test[:,:,:4])[0]

display_img_and_boxes(x_test[0,:,:,0], np.zeros((1,4)), gts[y_test[0,:,5]==1])

# plt.imshow(x_test[0,:,:,0])
# plt.show()
