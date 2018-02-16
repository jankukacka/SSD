#!/usr/bin/env python
# --------------------------------------------------------
# Generate dataset of sagittal projections and their bounding
# boxes, including various augmentations
# Copyright (c) 2017 Jan Kukacka
# --------------------------------------------------------

import numpy as np
import os
import scipy.stats
import scipy.ndimage.interpolation
import itertools
import argparse
import sys
from time import time
from skimage.transform import resize as skimg_resize
from scipy import interpolate
# --
import medpy.io.header
# --
dsdk_path = '/media/Data2/Jan/dataset-sdk/'
if dsdk_path not in sys.path:
    sys.path.append(dsdk_path)
import dataset_sdk as dsdk
import dataset_sdk.bbox
import dataset_sdk.io
# --

class DataAugmenter(object):
    def __init__(self, cts_root_path, **kwargs):
        '''
        kwargs: dictionary of extra parameters:
            o 'aggregation_plane': [0 or 'coronal', 1 or 'axial', 2 or 'sagittal' (Default)]
            o 'crop_scale': float. Passed to get_crop().
            o 'crop_max': float. Passed to get_crop().
            o 'use_crop': bool. If False, no cropping occurs. Default True.
            o 'aggregation_noise': float. Amount of noise on slice weights. Default 1.0.
            o 'aggregation_scale': float. How much should central slices be
                        preferred over sides. Lower numbers
                        mean more balanced weighting. Default 10.
            o 'aggregation_method': 'mean' (Default) or 'max'. Type of the
                        projection
            o 'rotation_angle_range': float. Rotation angle is selected uniformly
                        between +- this value. Default 5.0.
            o 'contrast': float. Image contrast may be enhanced. Value should
                        be in a range [0;0.1]. Default 0.
            o 'zmuv_mean': float. Zero-mean-unit-variance correction mean.
                        Not used if not present together with 'zmuv_std'.
            o 'zmuv_std': float. Zero-mean-unit-variance correction std. dev.
                        Not used if not present together with 'zmuv_mean'.

        '''
        ## Parse kwargs
        aug_settings_keys = ['crop_scale', 'crop_max', 'use_crop',
                             'aggregation_noise', 'aggregation_scale',
                             'rotation_angle_range', 'contrast', 'zmuv_mean',
                             'zmuv_std', 'aggregation_plane', 'aggregation_method']
        self.aug_settings = { key: kwargs[key] for key in aug_settings_keys if key in kwargs}
        settings = self.aug_settings
        if not 'use_crop' in settings: settings['use_crop'] = True
        if not 'aggregation_scale' in settings: settings['aggregation_scale'] = 10
        if not 'aggregation_noise' in settings: settings['aggregation_noise'] = 1
        if not 'rotation_angle_range' in settings: settings['rotation_angle_range'] = 5
        if not 'contrast' in settings: settings['contrast'] = 0
        if not 'aggregation_method' in kwargs: settings['aggregation_method'] = 'mean'
        if not 'aggregation_plane' in kwargs: settings['aggregation_plane'] = 2
        if settings['aggregation_plane'] == 'coronal':
            settings['aggregation_plane'] = 0
        if settings['aggregation_plane'] == 'axial':
            settings['aggregation_plane'] = 1
        if settings['aggregation_plane'] == 'sagittal':
            settings['aggregation_plane'] = 2
        if not settings['aggregation_plane'] in [0,1,2]:
            settings['aggregation_plane'] = 2
        ## -----
        self.cts_root_path = cts_root_path
        self.cache = {}

    def _get_image_from_memory(self, image_id):
        '''
        Checks if the image is already in the memory. If yes, returns it. If no,
        loads it and returns it.

        # Arguments
            image_id: int. CTS image id to load.
        # Returns: tuple.
            img: 3D image resampled to 1x1 sagittal isotropic resolution
            masks: see aggregate_label_masks
            labels: see aggregate_label_masks
        '''
        if not image_id in self.cache:
            img, img_header = dsdk.io.load_image(self.cts_root_path, image_id, True)
            seg = dsdk.io.load_segmentation(self.cts_root_path, image_id)
            img, seg = normalize_resolution(img, seg, img_header,
                                            self.aug_settings['aggregation_plane'])
            masks, labels = aggregate_label_masks(seg, axis=self.aug_settings['aggregation_plane'])
            self.cache[image_id] = (img, masks, labels)
        return self.cache[image_id]

    def get_image(self, image_id):
        '''
        '''
        img, masks, labels = self._get_image_from_memory(image_id)
        aug_img, bboxes = self._get_augmented_version(img, masks, labels)
        aug_img = aug_img.astype(np.float32)

        # restore bbox_info structure as defined in dataset_sdk.bbox
        bbox_info = { 'shape': np.array([aug_img.shape[0], aug_img.shape[1], 1]),
                      'slices': [{'bboxes': bboxes}] }
        return aug_img, bbox_info

    def _get_augmented_version(self, img, masks, labels):
        '''
        Generate augmented version of the given image and its mask

        Parameters:
            img: 3D image volume
            masks: [w*h*label_count] array with label masks
            labels: list of tuples contining index-to-label mapping for masks
        Returns:
            sag_img: [w*h] projected and augmented image
            bboxes: list of bounding boxes
        '''
        settings = self.aug_settings
        axis = settings['aggregation_plane']
        aggregation_method = settings['aggregation_method']

        # Generate weights
        x = xrange(img.shape[axis])
        scale = img.shape[axis]/settings['aggregation_scale']
        y = scipy.stats.norm.pdf(x, loc=img.shape[axis]/2, scale=scale)
        noise = np.random.normal(size=img.shape[axis], loc=0, scale=y/3)

        # aggregate sagittal projection
        if aggregation_method == 'mean':
            sag_img = np.average(img,
                                 axis=axis,
                                 weights=np.abs(y+settings['aggregation_noise']*noise))
        else:
            sag_img = np.max(img, axis=axis)

        # crop
        if settings['use_crop']:
            crop_settings = {}
            if 'crop_max' in settings:
                crop_settings['crop'] = settings['crop_max']
            if 'crop_scale' in settings:
                crop_settings['scale'] = settings['crop_scale']
            crop = get_crop(sag_img.shape, **crop_settings)

            sag_img = sag_img[crop[1][0]:crop[1][0]+crop[0][0]-1,
                              crop[1][1]:crop[1][1]+crop[0][1]-1]
            masks = masks[crop[1][0]:crop[1][0]+crop[0][0]-1,
                          crop[1][1]:crop[1][1]+crop[0][1]-1]

        # rotation
        angle = np.random.uniform(-settings['rotation_angle_range'], settings['rotation_angle_range'])
        sag_img = scipy.ndimage.interpolation.rotate(sag_img, angle=angle, cval=np.min(sag_img))
        masks = scipy.ndimage.interpolation.rotate(masks, angle=angle, order=0)
        bboxes = [dsdk.bbox._extract_from_slice((label[1])*masks[:,:,label[0]]) for label in labels]
        bboxes = itertools.chain.from_iterable(bboxes)

        if (settings['contrast'] > 0 and settings['contrast'] <= 0.1):
            co = np.random.normal(0, settings['contrast'])
            x = np.linspace(0, 1, num=5, endpoint=True)
            y = np.array([0, 0.25-co, 0.5, 0.75+co,1])
            tck = interpolate.splrep(x, y, s=0)

            mn = np.min(sag_img)
            mx = np.max(sag_img)
            normImg = (sag_img - mn) / (mx-mn+1e-10)
            sag_img = interpolate.splev(normImg, tck, der=0) * (mx-mn) + mn

        if 'zmuv_mean' in settings and 'zmuv_std' in settings:
            sag_img = (sag_img - settings['zmuv_mean']) / settings['zmuv_std']

        return sag_img, bboxes


def normalize_resolution(img, seg, hdr, axis=2):
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
    seg = skimg_resize(seg, newDims, preserve_range = True, order = 0)
    return img, seg

def aggregate_label_masks(seg, axis=2):
    '''
    Convert 3D volume of voxel labels into volume of 2D aggregated
    masks, one for each label present in the image.

    Parameters:
        seg: 3D array of voxel labels
        axis: [0,1,2]. Along which axis to aggregate. Default is 2.

    Returns:
        3D array of masks [w*h*label_count]
        list of tuples, containing index-to-label conversion of the returned array
    '''
    min_label = int(np.min(seg[seg>0]))
    max_label = int(np.max(seg))
    #print 'Labels range from {} to {}'.format(min_label, max_label)

    masks_shape = seg.shape[:axis] + seg.shape[axis+1:]
    masks = np.zeros(masks_shape+(max_label-min_label+1,), dtype=bool)
    labels = range(min_label, max_label+1)
    for i in enumerate(labels):
        masks[:,:,i[0]] = np.any(seg==i[1], axis=axis)
    return masks, list(enumerate(labels))

def get_crop(orig_shape, **settings):
    '''
    Generate a random crop for an image.

    Parameters:
        orig_shape: array-like with image dimensions
        kwargs - settings:
            scale: float, affects the distribution of cropping values. Default 0.95
            crop: float, minimum portion of image to pertain. Default 0.45

    Returns:
        ndarray of int with the new image shape
        ndarray of int with the offsets of the crop
    '''
    orig_shape = np.array(orig_shape)
    scale = settings['scale'] if 'scale' in settings else 0.95
    crop = settings['crop'] if 'crop' in settings else 0.45
    crop_ratio = 1+crop-np.maximum(scipy.stats.expon.pdf(np.random.rand(2), scale=scale)/scale, crop)
    new_shape = (orig_shape*crop_ratio).astype(int)
    offset_range = orig_shape - new_shape
    offset = np.random.uniform(np.zeros_like(offset_range), offset_range).astype(int)
    #print orig_shape, crop_ratio, new_shape, offset_range, offset
    return new_shape, offset
