# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Implementation of SSD models
# ------------------------------------------------------------------------------

from keras.layers import Conv2D, Reshape, Concatenate, Input, Activation
from keras.models import Model
from anchor_generator_layer import AnchorGeneratorLayer

def Simple_SSD(num_classes=4):
    '''

    # Arguments
        - num_classes: positive int, including background class
    '''
    net = {}
    net['input'] = x = Input((None, None, 1))
    net['conv1'] = x = Conv2D(64, (3,3), padding='same', activation='relu', name='conv1')(x)
    net['pool1'] = x = Conv2D(64, (2,2), padding='same', strides=(2,2), name='pool1')(x)
    net['conv2'] = x = Conv2D(128, (3,3), padding='same', activation='relu', name='conv2')(x)
    net['pool2'] = x = Conv2D(128, (2,2), padding='same', strides=(2,2), name='pool2')(x)
    net['conv3'] = x = Conv2D(256, (3,3), padding='same', activation='relu', name='conv3')(x)
    net['pool3'] = x = Conv2D(256, (2,2), padding='same', strides=(2,2), name='pool3')(x)
    net['conv4_1'] = x = Conv2D(512, (3,3), padding='same', activation='relu', name='conv4_1')(x)
    net['conv4_2'] = x = Conv2D(512, (3,3), padding='same', activation='relu', name='conv4_2')(x)
    net['pool4'] = x = Conv2D(512, (2,2), padding='same', strides=(2,2), name='pool4')(x)
    net['conv5_1'] = x = Conv2D(1024, (3,3), padding='same', activation='relu', name='conv5_1')(x)
    net['conv5_2'] = x = Conv2D(1024, (3,3), padding='same', activation='relu', name='conv5_2')(x)
    net['pool5'] = x = Conv2D(512, (2,2), padding='same', strides=(2,2), name='pool5')(x)
    net['conv6_1'] = x = Conv2D(256, (3,3), padding='same', activation='relu', name='conv6_1')(x)
    net['conv6_2'] = x = Conv2D(256, (3,3), padding='same', activation='relu', name='conv6_2')(x)

    # BBox prediction blocks
    # From conv5_2
    # anchors = [0.5, 1, 2]
    # net['bbox_conf_conv5_2'] = Conv2D(num_classes*len(anchors), (3,3), padding='same')(net['conv5_2'])
    # net['resh_bbox_conf_conv5_2'] = Reshape((-1,num_classes))(net['bbox_conf_conv5_2'])
    # net['softmax_bbox_conf_conv5_2'] = Activation('softmax')(net['resh_bbox_conf_conv5_2'])
    # net['bbox_loc_conv5_2']  = Conv2D(4*len(anchors), (3,3), padding='same')(net['conv5_2'])
    # net['resh_bbox_loc_conv5_2'] = Reshape((-1,4))(net['bbox_loc_conv5_2'])
    # net['anchor_conv5_2'] = AnchorGeneratorLayer(feature_stride=16, offset=55,
    #                                              aspect_ratios=anchors)(net['conv5_2'])

    # From conv6_2
    anchors = [0.5, 1]
    net['bbox_conf_conv6_2'] = Conv2D(num_classes*len(anchors), (3,3), padding='same')(net['conv6_2'])
    net['resh_bbox_conf_conv6_2'] = Reshape((-1,num_classes))(net['bbox_conf_conv6_2'])
    net['softmax_bbox_conf_conv6_2'] = Activation('softmax')(net['resh_bbox_conf_conv6_2'])
    net['bbox_loc_conv6_2']  = Conv2D(4*len(anchors), (3,3), padding='same')(net['conv6_2'])
    net['resh_bbox_loc_conv6_2'] = Reshape((-1,4))(net['bbox_loc_conv6_2'])
    net['anchor_conv6_2'] = AnchorGeneratorLayer(feature_stride=32, offset=0, #119
                                                 aspect_ratios=anchors, scale=2)(net['conv6_2'])

    # net['cat_loc'] = Concatenate(axis=1)([net['resh_bbox_loc_conv5_2'],
    #                                       net['resh_bbox_loc_conv6_2']])
    # net['cat_conf'] = Concatenate(axis=1)([net['softmax_bbox_conf_conv5_2'],
    #                                        net['softmax_bbox_conf_conv6_2']])
    # net['cat_anc'] = Concatenate(axis=1)([net['anchor_conv5_2'],
    #                                       net['anchor_conv6_2']])
    # net['output'] = Concatenate(axis=2)([net['cat_loc'],
    #                                      net['cat_conf'],
    #                                      net['cat_anc']])

    net['output'] = Concatenate(axis=2)([net['resh_bbox_loc_conv6_2'],
                                         net['softmax_bbox_conf_conv6_2'],
                                         net['anchor_conv6_2']])


    return Model(net['input'], net['output'])
