# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Implementation of Anchor box generator layer
# ------------------------------------------------------------------------------

import numpy as np
from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf # because of limited support in K.arange

class AnchorGeneratorLayer(Layer):
    '''
    Layer generating anchor bounding boxes for variable shaped input
    '''

    def __init__(self, feature_stride, offset, aspect_ratios=[1], scale=1, **kwargs):
        '''
        # Arguments
            - feature_stride: int. Determines stride of features at the input
                              layer.
            - offset: int. Determines how many pixels at each edge are lost
                      due to using 'valid' convolution mode. Set to 0 if
                      using 'same' mode.
            - aspect_ratios: iterable of aspect ratios (w/h) of anchor boxes
            - scale: positive float. Determines the scale of the bounding boxes.
                     If scale == 1, bboxes have width==feature_stride
        '''

        self.aspect_ratios = aspect_ratios[:]   # create a copy of the list
        self.feature_stride = feature_stride
        self.offset = offset
        self.scale = scale
        super(AnchorGeneratorLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        '''
        # Arguments
            - input_shape: tuple of int: (batch_size, w, h, channels).
                           Expecting tensorflow data ordering.
        # Returns
            - Output shape: tuple of int: (batch_size, w*h*anchor_count, 4).
        '''
        anchor_count = len(self.aspect_ratios)
        batch_size = input_shape[0]
        width = input_shape[1]
        height = input_shape[2]
        return (None, None, 4)

    def call(self, x):
        '''
        Generate tensor of anchor bounding boxes in the format:
        (center_x, center_y, width, height)

        At each point of the feature maps creates anchor boxes with specified
        aspect ratios. In total creates width*height*anchor_count bounding boxes
        '''

        input_shape = K.shape(x)
        batch_size = input_shape[0]
        width = input_shape[1]
        height = input_shape[2]
        # following two things simulate meshgrid(xrange(height), xrange(width))
        anchors_y = K.tile(tf.range(height), [width]) # replace tf.range for K.arange
                                             # once tensor input is supported
                                             # same in the following:
        anchors_x = K.reshape(
                        K.permute_dimensions(
                            K.tile(
                                K.expand_dims(tf.range(width), axis=0),
                                (height,1)),
                            (1,0)),
                        (width*height,))
        anchors_centers = K.permute_dimensions(
                            K.stack([anchors_y, anchors_x], axis=0),
                            (1,0))
        anchors_centers = (anchors_centers * self.feature_stride
                           + self.feature_stride / 2
                           + self.offset)

        anchors_centers = K.reshape(K.tile(anchors_centers, (1, len(self.aspect_ratios))), (-1,2))
        anchors_centers = K.cast(anchors_centers, 'float32')

        ar = np.array(self.aspect_ratios)
        size = self.feature_stride * self.scale
        widths = size*ar
        heights = size/ar
        sizes = K.variable(np.transpose(np.vstack((widths, heights)), (1,0)))

        sizes = K.reshape(sizes, (1,-1))
        sizes = K.tile(sizes, (1,width*height))
        sizes = K.reshape(sizes, (-1,2))

        #anchors_tensor = K.reshape(K.stack((anchors_centers, sizes), axis=1), (-1,4))
        anchors_tensor = K.concatenate((anchors_centers, sizes), axis=-1)
        anchors_tensor = K.expand_dims(anchors_tensor, 0)
        pattern = (batch_size, 1, 1)
        anchors_tensor = K.tile(anchors_tensor, pattern)
        return anchors_tensor


class AnchorGenerator(object):
    '''
    Object generating anchor bounding boxes
    '''

    def __init__(self, feature_stride, offset, aspect_ratios=[1], scale=1):
        '''
        # Arguments
            - feature_stride: int. Determines stride of features at the input
                    layer.
            - offset: int. Determines how many pixels at each edge are lost
                    due to using 'valid' convolution mode. Set to 0 if
                    using 'same' mode.
            - aspect_ratios: iterable of aspect ratios (w/h) of anchor boxes
            - scale: positive float or a list of positive floats. Determines the
                    scale of the bounding boxes. If scale == 1, bboxes have
                    width==feature_stride
        '''

        self.aspect_ratios = aspect_ratios[:]   ## create a copy of the list
        self.feature_stride = feature_stride
        self.offset = offset
        ## Ensure the scale is in a list
        try:
            self.scale = [s for s in scale]
        except:
            self.scale = [scale]

    def Generate(self, input_shape):
        '''
        Generate anchor boxes for the batch of the given shape. At each point of
        the feature maps creates anchor boxes with specified aspect ratios.
        In total creates (width/feature_stride)*(height/feature_stride)*anchor_count bounding boxes

        # Arguments
            - input_shape. Tuple of positive ints. Shape of the input batch.
                           (batch_size, width, height, channels)
        # Returns
            - anchors. numpy array of shape (num_anchors, 5) encoded as
                       [center_x, center_y, width, height, valid_bit]
                       valid_bit == 1 if the whole bbox is inside the image
        '''
        batch_size = input_shape[0]
        width = input_shape[1] // self.feature_stride
        height = input_shape[2] // self.feature_stride
        anchors_y, anchors_x = np.meshgrid(xrange(height), xrange(width))

        anchors_centers = np.transpose(
                            np.vstack([anchors_y.flatten(), anchors_x.flatten()]),
                            (1,0))
        anchors_centers = (anchors_centers * self.feature_stride
                           + self.feature_stride / 2
                           + self.offset)

        anchors_centers = np.reshape(np.tile(anchors_centers,
            (1, len(self.aspect_ratios) * len(self.scale))), (-1,2))
        anchors_centers = anchors_centers.astype('float32')

        ar = np.array(self.aspect_ratios)
        size = self.feature_stride # * self.scale
        widths = size*ar
        heights = size/ar
        sizes = np.transpose(np.vstack((widths, heights)), (1,0))

        ## Duplicate for each scale
        sizes = np.reshape(np.repeat(sizes[np.newaxis,:,:], len(self.scale), axis=0), (-1,2))
        scales = np.repeat(self.scale, len(self.aspect_ratios))
        sizes = sizes * scales[:, np.newaxis]

        sizes = np.reshape(sizes, (1,-1))
        sizes = np.tile(sizes, (1,width*height))
        sizes = np.reshape(sizes, (-1,2))

        anchors_tensor = np.concatenate((anchors_centers, sizes), axis=-1)
        anchors_tensor = np.expand_dims(anchors_tensor, 0)
        pattern = (batch_size, 1, 1)
        anchors_tensor = np.tile(anchors_tensor, pattern)
        return anchors_tensor
