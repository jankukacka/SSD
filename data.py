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
# --
import random
# --
from data_augmenter import DataAugmenter

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
                 padding=0, min_voxels=500,max_images=-1, use_two_classes=False):
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
            - settings: kwargs for DataAugmenter. For details see DataAugmenter.
        '''
        self.batch_size = batch_size
        self.padding = padding
        self.min_voxels = min_voxels

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


    def get_augmented_img(self, image_id, depth=0):
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
            return self.get_augmented_img(image_id, depth+1)
        s = img.shape
        if 1.*s[0]/s[1] < 0.3 or \
           1.*s[1]/s[0] < 0.3: # originally .15
           ## Bad w/h ratio
           return self.get_augmented_img(image_id, depth+1)

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
                for j in xrange(len(targets[i]['slices'][0]['bboxes'])):
                    target_tensor[i,j, :5] = dsdk.bbox.bbox_to_ccwhl(bboxes[j])
                    target_tensor[i,j, :2] += pad # adjust for padding
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
