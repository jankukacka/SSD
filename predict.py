# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 2/2018
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Prediction routine
# ------------------------------------------------------------------------------


# --
import medpy.io
import numpy as np
import cPickle
from math import sqrt
# --
from lib.net import Residual_SSD
from lib.anchor_generator_layer import AnchorGenerator
from lib.utils import aggregate_bboxes_ccwh
from skimage.transform import resize as skimg_resize
# --

def predict(filename):
    '''
    Provides bounding box prediction for the given file
    '''
    img, img_header = medpy.io.load(filename)

    model = Residual_SSD(num_classes=2, use_bn=True,
                         num_anchors=4)

    sagittal = predict_projection('sagittal', img, img_header, model)
    print 'sagittal', sagittal
    coronal = predict_projection('coronal', img, img_header, model)
    print 'coronal', coronal

    res = np.array(medpy.io.header.get_pixel_spacing(img_header))
    s = np.array(img.shape)*res

    corner1 = (max(0,sagittal[1]-.5*sagittal[3]),
               max(0,
                   min(sagittal[0]-.5*sagittal[2],
                       coronal[1]-.5*coronal[3])),
               max(0,coronal[0]-.5*coronal[2]))

    corner2 = (min(s[0],sagittal[1]+.5*sagittal[3]),
               min(s[1],
                   max(sagittal[0]+.5*sagittal[2],
                       coronal[1]+.5*coronal[3])),
               min(s[2],coronal[0]+.5*coronal[2]))

    print 'corner1', corner1
    print 'corner2', corner2

    c1 = (np.array(corner1) / res).astype(np.int)
    c2 = (np.array(corner2) / res).astype(np.int)
    return c1, c2



def normalize_resolution(img, hdr, axis=2):
    '''
    Normalize image sagittal resolution to isotropic 1x1mm

    # Arguments
        - axis = [0,1,2] -  axis whose resolution should be preserved. Default 2.
    '''
    # Get real voxel dimensions
    res = np.array(medpy.io.header.get_pixel_spacing(hdr))
    # Get image dimensions
    imgDims = np.array(img.shape)
    # Set new image resolution
    newRes = np.ones(3)
    newRes[axis] = res[axis]
    # Set new image dimensions
    voxelRatio = newRes / res
    newDims = np.floor(imgDims / voxelRatio).astype(int)
    img = skimg_resize(img, newDims, preserve_range = True)
    return img

def pred2bbox(anchor, pred):
    centers = anchor[:,2:]*pred[:,:2] + anchor[:,:2]
    sizes = anchor[:,2:]*np.exp(pred[:,2:])
    return np.concatenate((centers, sizes), axis=-1)

def predict_projection(aggregation_plane, img, img_header, model):
    '''
    Provides prediction in a single plane
    '''

    snapshot = '/media/Data2/Jan/ssd-keras/ssd_keras_my/output/residual_ssd/cts_{}_max_train_spine/epoch_150.pkl'.format(aggregation_plane)
    aug_settings = {
        'use_crop': False,
        'aggregation_plane': aggregation_plane,
        'aggregation_plane_index': 0 if aggregation_plane == 'coronal' else 2,
        'aggregation_method': 'max',
        'aggregation_noise': 0,
        'rotation_angle_range': 0,
        'zmuv_std': 350.0,#np.std(img), #350.0,      # approx from other datasets
        'zmuv_mean': 200#np.mean(img)
    }

    if aggregation_plane == 'coronal':
        aspect_ratios = [sqrt(.2), sqrt(.4)]
        scales = (5,6.5)
        min_wh_ratio=.05
    else:
        aspect_ratios = [sqrt(2.5), sqrt(3.5)]
        scales = (5,7.5)
        min_wh_ratio=.3

    ag = AnchorGenerator(feature_stride=32, offset=0, scale=scales,
                         aspect_ratios=aspect_ratios)


    img = normalize_resolution(img, img_header, aug_settings['aggregation_plane_index'])
    img2d = np.max(img, axis=aug_settings['aggregation_plane_index'])
    img2d = (img2d - aug_settings['zmuv_mean']) / aug_settings['zmuv_std']


    s = np.array(img2d.shape)
    max_dims = ((s + 31) / 32) * 32 # round up to 32-divisible dimension

    input_tensor = np.zeros((1, max_dims[0], max_dims[1],1))

    pad = (max_dims - s) // 2 # pad around all edges
    input_tensor[0] = img2d.min() # pad image with the lowest value (simulating air)
    input_tensor[0, pad[0]:pad[0]+s[0], pad[1]:pad[1]+s[1], 0] = img2d

    ## Generate anchors for this batch
    anchors = ag.Generate(input_tensor.shape)
    anchors = anchors[0]


    with open(snapshot, 'rb') as f:
        w = cPickle.load(f)
    model.set_weights(w)

    pred = model.predict(input_tensor, batch_size=1)
    pred = pred[0]

    cur_anchors = np.copy(anchors)
    cur_anchors[:,:2] -= pad[::-1]
    crop_preds = pred2bbox(cur_anchors,pred[:,:4])
    positives = pred[:,5] > .90
    if np.sum(positives) < 2:
        # too few positives, use top_k method
        positives = np.argsort(-np.max(pred[:,5:], axis=-1))[:5] # k = 5


    aggregated = aggregate_bboxes_ccwh(crop_preds[positives,:4], 'mean')
    return aggregated
