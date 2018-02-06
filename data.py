# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Data preparation
# ------------------------------------------------------------------------------

# --
import numpy as np
import os
import random
# --
from math import sqrt
# --
from data_augmenter import DataAugmenter
from anchor_generator_layer import AnchorGenerator
from anchor_assignment import Match

import dataset_sdk as dsdk
import dataset_sdk.bbox
import dataset_sdk.imageset
# --

class DataGenerator(object):
    '''
    Data generator class.
    Loads a dataset and yields batches in Generate function.
    Currently works with sagittal projections augmented dataset.
    '''

    def __init__(self, batch_size, folder_path, padding=0, min_voxels=500,
                 max_images=-1, use_two_classes=False):
        '''
        Initializes the data generator.

        # Arguments:
            - batch_size: positive int. number of samples per batch.
            - folder_path: string. path to the folder with the data.
            - padding: positive int. Number of pixels to use as padding around
                       GT bounding boxes. Default 0.
            - min_voxels: positive int. Minimum number of pixels to consider a
                          bounding box. Smaller ones are ignored. Default 500.
            - max_images: int. Limits the number of images to use for the
                          generator. If max_images < 0 or
                          max_images > total_images, all images will be used.
        '''
        self.batch_size = batch_size
        # 3 vertebra categories
        if not use_two_classes:
            self.classes = ('__background__', # always index 0
                             'cervical', 'thoracic', 'lumbar')
        else:
            self.classes = ('__background__', 'vetebra')

        # Load dataset
        print 'Loading dataset', folder_path
        self.images = np.load(os.path.join(folder_path, 'images.npz'))
        bbox_slices = np.load(os.path.join(folder_path, 'bbox_slices.npz'))
        bbox_metadata = np.load(os.path.join(folder_path, 'bbox_metadata.npz'))
        print 'Finished.'

        print 'Preparing bounding boxes...'
        limit = len(self.images.keys()) if max_images < 0 else max_images
        self.image_index = self.images.keys()[:min(len(self.images.keys()),limit)]
        bboxes = {}
        for image in self.image_index:
            bboxes[image] = dsdk.bbox.numpy_to_bbox_info(bbox_metadata[image],
                                                         bbox_slices[image])

        # prepare bounding boxes
        self.filtered_bboxes = {image: dsdk.bbox.filter_bbox_info(bboxes[image],
                   drop_empty=True,
                   padding=padding,
                   min_voxels=min_voxels) for image in self.image_index}

        # remove images with bad shape (too narrow, too wide)
        to_remove = []
        for image in self.image_index:
            if self.filtered_bboxes[image]['slice_count'] == 0:
                to_remove.append(image)
                continue
            s = self.images[image].shape
            if 1.*s[0]/s[1] < 0.3 or \
               1.*s[1]/s[0] < 0.3: # originally .15
               to_remove.append(image)
        for image in to_remove:
            self.image_index.remove(image)
        print 'Finished.'

        if len(self.classes) == 2:
            ## Convert classes to 2 class case
            for image in self.image_index:
                bbox_info = self.filtered_bboxes[image]
                for sl in bbox_info['slices']:
                    for bbox in sl['bboxes']:
                        bbox['class'] = 1

        self.steps_per_epoch = len(self.image_index) // batch_size


    def Generate(self, shuffle=True):
        print 'Warning: This generator generates data in the old format (ground truth boxes only).'
        print 'To get anchor offsets, use OnlineDataGenerator instead.'
        num_keys = len(self.image_index)

        def _prepare_batch(inputs, targets):
            max_dims = np.array([512,512])
            max_bboxes = 0

            for i in xrange(self.batch_size):
                max_dims = np.maximum(max_dims, inputs[i].shape)
                num_bboxes =  len(targets[i]['slices'][0]['bboxes'])
                max_bboxes = max(num_bboxes, max_bboxes)

            input_tensor = np.zeros((self.batch_size, max_dims[0], max_dims[1],1))
            # zero padded input for some reason
            target_tensor = np.zeros((self.batch_size, max_bboxes, 8+len(self.classes)))

            for i in xrange(self.batch_size):
                s = inputs[i].shape
                input_tensor[i] = inputs[i].min() # pad image with the lowest value (simulating air)
                input_tensor[i, :s[0], :s[1], 0] = inputs[i]

                bboxes = targets[i]['slices'][0]['bboxes']
                for j in xrange(len(targets[i]['slices'][0]['bboxes'])):
                    target_tensor[i,j, :5] = dsdk.bbox.bbox_to_ccwhl(bboxes[j])
            return input_tensor, target_tensor

        targets = []
        inputs = []
        while True:
            if shuffle:
                random.shuffle(self.image_index)
            keys = self.image_index
            for key in keys:
                targets.append(self.filtered_bboxes[key])
                inputs.append(self.images[key])

                if len(targets) == self.batch_size:
                    yield _prepare_batch(inputs, targets)
                    inputs = []
                    targets = []

class OnlineDataGenerator(object):
    '''
    Data generator class.
    Loads a dataset and yields batches in Generate function.
    Unlike the DataGenerator, OnlineDataGenerator performs data augmentation
    on-the-fly. Generates sagittal projections.
    '''

    def __init__(self, batch_size, imageset_name, cts_root_path, settings,
                 padding=0, min_voxels=500,max_images=-1, use_two_classes=False,
                 return_anchors=False, anchor_generator=None, overlap_threshold=.5,
                 match_anchors=True, min_wh_ratio=0.3):
        '''
        Initializes the data generator.

        # Arguments:
            - batch_size: positive int. number of samples per batch.
            - imageset_name: string. Name of the imageset to use.
            - cts_root_path: string. Path to the root folder of the cts dataset.
            - padding: positive int. Number of pixels to use as padding around
                        GT bounding boxes. Default 0.
            - min_voxels: positive int. Minimum number of pixels to consider a
                        bounding box. Smaller ones are ignored. Default 500.
            - max_images: int. Limits the number of images to use for the
                        generator. If max_images < 0 or
                        max_images > total_images, all images will be used.
            - return_anchors: bool. If true, generator returns the anchors along
                        with the offsets as the last four dimensions.
                        Default False.
            - use_two_classes: bool. If true, generator converts the labels of
                        the vertebrae from [0..3] to [0,1]. Default False.
            - anchor_generator: AnchorGenerator instance or a list thereof.
                        If None (default), a generator with default settings
                        is used. If supplying a list, they must be in the same
                        order as predictions are concatenated in the used net.
            - settings: kwargs for DataAugmenter. For details see DataAugmenter.
            - overlap_threshold: float in range [0;1]. Minimum threshold to
                        consider an anchor to be responsible for a GT box.
            - match_anchors: bool. Default True. If True, generates data as numpy
                        arrays of anchors and their desired offsets. If False,
                        generates data in the same format as the DataGenerator,
                        i.e. array of gt bboxes.
            - min_wh_ratio: positive float. Smallest width/height or height/width
                        ratio of generated image to be accepted. Default 0.3.
        '''
        self.batch_size = batch_size
        self.padding = padding
        self.min_voxels = min_voxels
        self.return_anchors = return_anchors
        self.overlap_threshold = overlap_threshold
        self.match_anchors = match_anchors
        self.min_wh_ratio = min_wh_ratio

        # 3 vertebra categories
        if not use_two_classes:
            self.classes = ('__background__', # always index 0
                             'cervical', 'thoracic', 'lumbar')
        else:
            self.classes = ('__background__', 'vetebra')

        # Load dataset
        self.imageset_list = dsdk.imageset.load_imageset_by_name(cts_root_path, imageset_name)
        limit = len(self.imageset_list) if max_images < 0 else max_images
        self.imageset_list = self.imageset_list[:min(len(self.imageset_list),limit)]

        self.steps_per_epoch = len(self.imageset_list) // batch_size

        self.augmenter = DataAugmenter(cts_root_path, **settings)
        if anchor_generator is None:
            self.anchor_generator = AnchorGenerator(feature_stride=32,
                                                    offset=0,
                                                    aspect_ratios=[sqrt(0.5), 1],
                                                    scale=2)
        else:
            self.anchor_generator = anchor_generator


    def get_augmented_img(self, image_id, depth=0):
        '''
        Returns an augmented version of an image from the ct-spine dataset.

        # Arguments
            - image_id: positive int. int-id of the desired image.
            - depth: internal flag for stopping recursion in case a valid image
                    generation fails.

        # Returns
            - img: numpy array of shape (height, width)
            - bbox_info: bbox_info dictionary. For more info see dataset_sdk.bbox

            If method fails to generate a valid image, it prints a message and
            returns (None, None) tuple.
        '''
        if depth > 5:
            print 'Could not generate valid input from image', image_id
            return None, None

        img, bbox_info = self.augmenter.get_image(image_id)
        # post-process and validate
        bbox_info = dsdk.bbox.filter_bbox_info(bbox_info,
                   drop_empty=True,
                   padding=self.padding,
                   min_voxels=self.min_voxels)

        ## Invalid images:
        if bbox_info['slice_count'] == 0:
            ## No bboxes
            return OnlineDataGenerator.get_augmented_img(self,image_id, depth+1)
        s = img.shape
        if 1.*s[0]/s[1] < self.min_wh_ratio or \
           1.*s[1]/s[0] < self.min_wh_ratio:
           ## Bad w/h ratio
           return OnlineDataGenerator.get_augmented_img(self,image_id, depth+1)

        ## all good!
        if len(self.classes) == 2:
            ## Convert classes to 2 class case
            for sl in bbox_info['slices']:
                for bbox in sl['bboxes']:
                    bbox['class'] = 1
        return img, bbox_info

    def Generate(self, shuffle=True):
        num_keys = len(self.imageset_list)

        def _prepare_batch(inputs, targets):
            max_dims = np.array([32,32])
            max_bboxes = 0

            for i in xrange(self.batch_size):
                max_dims = np.maximum(max_dims, inputs[i].shape)
                num_bboxes =  len(targets[i]['slices'][0]['bboxes'])
                max_bboxes = max(num_bboxes, max_bboxes)

            max_dims = ((max_dims + 31) / 32) * 32 # round up to 32-divisible dimension

            input_tensor = np.zeros((self.batch_size, max_dims[0], max_dims[1],1))
            # zero padded input for some reason
            target_tensor = np.zeros((self.batch_size, max_bboxes, 8+len(self.classes)))

            for i in xrange(self.batch_size):
                s = np.array(inputs[i].shape)
                pad = (max_dims - s) // 2 # pad around all edges
                input_tensor[i] = inputs[i].min() # pad image with the lowest value (simulating air)
                input_tensor[i, pad[0]:pad[0]+s[0], pad[1]:pad[1]+s[1], 0] = inputs[i]

                bboxes = targets[i]['slices'][0]['bboxes']
                for j in xrange(len(bboxes)):
                    target_tensor[i,j, :5] = dsdk.bbox.bbox_to_ccwhl(bboxes[j])
                    ## Adjust labels for spine bounding boxes
                    if 'is_spine' in bboxes[j] and bboxes[j]['is_spine']:
                        target_tensor[i,j,4] = 1
                    target_tensor[i,j, :2] += pad[::-1] # adjust for padding

            ## Old data format, now mostly for debugging
            if not self.match_anchors:
                return input_tensor, target_tensor

            ## Generate anchors for this batch
            anchors = self.anchor_generator.Generate(input_tensor.shape)
            ## Match anchors to the ground truth boxes
            target_tensor = Match(target_tensor, anchors, len(self.classes),
                                  self.overlap_threshold,
                                  (input_tensor.shape[2],input_tensor.shape[1]))

            ## This includes also anchors in the data
            if self.return_anchors:
                target_tensor = np.concatenate((target_tensor, anchors), axis=-1)

            return input_tensor, target_tensor

        targets = []
        inputs = []
        while True:
            if shuffle:
                random.shuffle(self.imageset_list)
            for image_id in self.imageset_list:
                img, bbox_info = self.get_augmented_img(image_id)
                if img is None:
                    print 'Skipping an image', image_id
                    ## In case no valid image could be generated from this sample...
                    continue
                targets.append(bbox_info)
                inputs.append(img)

                if len(targets) == self.batch_size:
                    yield _prepare_batch(inputs, targets)
                    inputs = []
                    targets = []

class OnlineSpineDataGenerator(OnlineDataGenerator):
    '''
    Data generator for whole spine bounding box predictions.
    Derives from OnlineDataGenerator, just automatically converts bounding boxes
    to one spine bbox.
    '''

    def __init__(self, **kwargs):
        '''
        For kwargs see OnlineDataGenerator args. They are passed directly through.
        '''
        super(OnlineSpineDataGenerator, self).__init__(**kwargs)
        self.classes = ('__background__', # always index 0
                        'spine')

    def get_augmented_img(self, image_id):
        '''
        Like OnlineDataGenerator.get_augmented_img, but converts generated bboxes
        to one large spine bbox.
        '''

        img, bbox_info = super(OnlineSpineDataGenerator, self).get_augmented_img(image_id)
        ## If parent method fails, pass the negative result
        if img is None:
            return None, None
        ## Extract spine bbox
        bbox_info = dsdk.bbox.bbox_info_to_spine_bbox(bbox_info)

        return img, bbox_info
