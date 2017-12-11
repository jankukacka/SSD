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
import dataset_sdk as dsdk
import dataset_sdk.bbox
# --

class DataGenerator(object):
    '''
    Data generator class.
    Loads a dataset and yields batches in Generate function.
    Currently works with sagittal projections augmented dataset.
    '''

    def __init__(self, batch_size, folder_path, padding=0, min_voxels=500):
        '''
        Initializes the data generator.

        # Arguments:
            - batch_size: positive int. number of samples per batch.
            - folder_path: string. path to the folder with the data.
            - padding: positive int. Number of pixels to use as padding around
                       GT bounding boxes. Default 0.
            - min_voxels: positive int. Minimum number of pixels to consider a
                          bounding box. Smaller ones are ignored. Default 500.
        '''
        self.batch_size = batch_size
        # 3 vertebra categories
        self.classes = ('__background__', # always index 0
                         'cervical', 'thoracic', 'lumbar')

        # Load dataset
        print 'Loading dataset', folder_path
        self.images = np.load(os.path.join(folder_path, 'images.npz'))
        bbox_slices = np.load(os.path.join(folder_path, 'bbox_slices.npz'))
        bbox_metadata = np.load(os.path.join(folder_path, 'bbox_metadata.npz'))
        print 'Finished.'

        print 'Preparing bounding boxes...'
        self.image_index = self.images.keys()[:min(len(self.images.keys()),1000)]
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
            target_tensor = np.zeros((self.batch_size, max_bboxes, 12))

            for i in xrange(self.batch_size):
                s = inputs[i].shape
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
