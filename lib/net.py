# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Implementation of SSD models
# ------------------------------------------------------------------------------

# --
import tensorflow as tf
# --
from math import sqrt
# --
from keras.layers import Conv2D, Reshape, Concatenate, Input, Activation, MaxPooling2D, Add, AveragePooling2D, BatchNormalization
from keras.models import Model
# --
from anchor_generator_layer import AnchorGeneratorLayer
# --

def _get_conv(num_filters, kernel_shape, use_bn, name, x, use_relu=True, strides=(1,1)):
    if use_bn:
        x = Conv2D(num_filters, kernel_shape, padding='same', name=name, use_bias=False, strides=strides)(x)
        x = BatchNormalization()(x)
        if use_relu:
            x = Activation('relu')(x)
    else:
        if use_relu:
            x = Conv2D(num_filters, kernel_shape, padding='same', activation='relu', name=name, strides=strides)(x)
        else:
            x = Conv2D(num_filters, kernel_shape, padding='same', name=name, strides=strides)(x)
    return x

def Residual_SSD(num_classes=4, use_bn=False, num_anchors=2):
    '''

    # Arguments
        - num_classes: positive int, including background class
        - num_anchors: positive int. Number of anchor boxes per location.
                Default = 2.
    '''
    net = {}
    net['input'] = x = Input((None, None, 1))
    with tf.name_scope('conv_block_1'):
        net['conv1'] = x = _get_conv(64, (3,3), use_bn, 'conv1', x)
        net['pool1'] = x = MaxPooling2D(pool_size=(2,2), padding='same', name='pool1')(x)

    use_bn = False
    # Residual block 1
    with tf.name_scope('res_block_1'):
        net['conv2_1'] = x = _get_conv(128, (3,3), use_bn, 'conv2_1', x)
        net['conv2_2'] = x = _get_conv(128, (3,3), use_bn, 'conv2_2', x, False)
        net['pool1_dup'] = Concatenate(axis=-1, name='pool1_dup')([net['pool1'], net['pool1']])
        net['sum2'] = x = Add()([x, net['pool1_dup']])
        net['relu2'] = x = Activation('relu')(x)

    # Residual block 2
    with tf.name_scope('res_block_2'):
        net['pool2'] = x = _get_conv(128, (2,2), use_bn, 'pool2', x, strides=(2,2))
        net['conv3'] = x = _get_conv(256, (3,3), use_bn, 'conv3', x, False)
        net['res2_pool'] = AveragePooling2D(pool_size=(2,2), padding='same')(net['relu2'])
        net['res2_pool_dup'] = Concatenate(axis=-1)([net['res2_pool'],net['res2_pool']])
        net['sum3'] = x = Add()([x, net['res2_pool_dup']])
        net['relu3'] = x = Activation('relu')(x)

    # Residual block 3
    with tf.name_scope('res_block_3'):
        net['pool3'] = x = _get_conv(256, (2,2), use_bn, 'pool3', x, strides=(2,2))
        net['conv4_1'] = x = _get_conv(512, (3,3), use_bn, 'conv4_1', x, False)
        net['res3_pool'] = AveragePooling2D(pool_size=(2,2), padding='same')(net['relu3'])
        net['res3_pool_dup'] = Concatenate(axis=-1)([net['res3_pool'],net['res3_pool']])
        net['sum4'] = x = Add()([x, net['res3_pool_dup']])
        net['relu4'] = x = Activation('relu')(x)

    # Residual block 4
    with tf.name_scope('res_block_4'):
        net['conv4_2'] = x =  _get_conv(512, (3,3), use_bn, 'conv4_2', x)
        net['conv4_3'] = x =  _get_conv(512, (3,3), use_bn, 'conv4_3', x, False)
        net['sum5'] = x = Add()([x, net['relu4']])
        net['relu5'] = x = Activation('relu')(x)

    # Residual block 5
    with tf.name_scope('res_block_5'):
        net['pool4'] = x = _get_conv(512, (2,2), use_bn, 'pool4', x, strides=(2,2))
        net['conv5_1'] = x = _get_conv(1024, (3,3), use_bn, 'conv5_1', x, False)
        net['res5_pool'] = AveragePooling2D(pool_size=(2,2), padding='same')(net['relu5'])
        net['res5_pool_dup'] = Concatenate(axis=-1)([net['res5_pool'],net['res5_pool']])
        net['sum6'] = x = Add()([x, net['res5_pool_dup']])
        net['relu6'] = x = Activation('relu')(x)

    # Residual block 6
    with tf.name_scope('res_block_6'):
        net['conv5_2'] = x = _get_conv(1024, (3,3), use_bn, 'conv5_2', x)
        net['conv5_3'] = x = _get_conv(1024, (3,3), use_bn, 'conv5_3', x, False)
        net['sum7'] = x = Add()([x, net['relu6']])
        net['relu7'] = x = Activation('relu')(x)

    # Residual block 7
    with tf.name_scope('res_block_7'):
        def my_init(shape, dtype=None):
            # print shape
            import numpy as np
            w = np.eye(256) * .25
            w = np.repeat(w,4,axis=0)
            w = np.reshape(w,(1,1,1024,256))
            return w

        net['pool5'] = x = _get_conv(512, (2,2), use_bn, 'pool5', x, strides=(2,2))
        net['conv6_1'] = x =  _get_conv(256, (3,3), use_bn, 'conv6_1', x, False)
        net['res7_pool'] = AveragePooling2D(pool_size=(2,2), padding='same')(net['relu7'])
        net['res7_pool_red'] = Conv2D(256, (1,1), name='res7_pool_red', use_bias=False, kernel_initializer=my_init, trainable=False)(net['res7_pool'])
        net['sum8'] = x = Add()([x, net['res7_pool_red']])
        net['relu8'] = x = Activation('relu')(x)

    with tf.name_scope('res_block_8'):
        net['conv6_2'] = x =  _get_conv(256, (3,3), use_bn, 'conv6_2', x)
        net['conv6_3'] = x =  _get_conv(256, (3,3), use_bn, 'conv6_3', x, False)
        net['sum9'] = x = Add()([x, net['relu8']])
        net['relu9'] = x = Activation('relu')(x)


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
    with tf.name_scope('bbox_predictor_2'):
        net['bbox_conf_relu9'] = Conv2D(num_classes*num_anchors, (3,3), padding='same', name='relu9_bbox_conf')(net['relu9'])
        net['resh_bbox_conf_relu9'] = Reshape((-1,num_classes))(net['bbox_conf_relu9'])
        net['softmax_bbox_conf_relu9'] = Activation('softmax')(net['resh_bbox_conf_relu9'])
        net['bbox_loc_relu9']  = Conv2D(4*num_anchors, (3,3), padding='same', name='relu9_bbox_loc')(net['relu9'])
        net['resh_bbox_loc_relu9'] = Reshape((-1,4))(net['bbox_loc_relu9'])
        # net['anchor_relu9'] = AnchorGeneratorLayer(feature_stride=32, offset=0, #119
        #                                              aspect_ratios=anchors, scale=2)(net['relu9'])

    # net['cat_loc'] = Concatenate(axis=1)([net['resh_bbox_loc_conv5_2'],
    #                                       net['resh_bbox_loc_conv6_2']])
    # net['cat_conf'] = Concatenate(axis=1)([net['softmax_bbox_conf_conv5_2'],
    #                                        net['softmax_bbox_conf_conv6_2']])
    # net['cat_anc'] = Concatenate(axis=1)([net['anchor_conv5_2'],
    #                                       net['anchor_conv6_2']])
    # net['output'] = Concatenate(axis=2)([net['cat_loc'],
    #                                      net['cat_conf'],
    #                                      net['cat_anc']])

    net['output'] = Concatenate(axis=2)([net['resh_bbox_loc_relu9'],
                                         net['softmax_bbox_conf_relu9']
                                         ])
                                         # ,
                                         # net['anchor_relu9']])


    return Model(net['input'], net['output'])
