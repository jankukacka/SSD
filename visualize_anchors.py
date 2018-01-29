import numpy as np
# import matplotlib
# matplotlib.set_backend
import matplotlib.pyplot as plt
from math import sqrt
from anchor_generator_layer import AnchorGenerator
from anchor_assignment import Match, _iou, _assign_boxes
from data import DataGenerator


data_gen = DataGenerator(batch_size=2, folder_path='/media/Data2/Jan/py-faster-rcnn/data/sagittal_projections/cts_train_large', max_images=2, use_two_classes=True)
ag = AnchorGenerator(feature_stride=32, offset=0, aspect_ratios=[sqrt(0.5), 1], scale=1.5)
data = next(data_gen.Generate())

shape = data[0][0].shape
anchors = ag.Generate((1, shape[0], shape[1], shape[2]))
bboxes = np.expand_dims(data[1][0], 0)
offsets = Match(bboxes, anchors, 2, .3, (shape[1], shape[0]))

#print offsets[offsets[:,:,5] == 1, :4]

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
negatives = single_image[bottom]
#negatives = negatives[np.random.choice(negatives.shape[0], 3*np.sum(top), replace=False)]
#display_img_and_boxes(img, single_image[top], data[1][0])
display_img_and_boxes(img, anchors[0,1::8], None)
display_img_and_boxes(img, anchors[0][top], data[1][0])
