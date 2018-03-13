# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Implementation of Anchor box generator
# ------------------------------------------------------------------------------

import numpy as np

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
