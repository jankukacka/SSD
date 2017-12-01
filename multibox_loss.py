# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Implementation of Multibox loss
# ------------------------------------------------------------------------------

import keras.backend as K
import numpy as np
import tensorflow as tf

def _iou(bboxes, anchors):
    '''
    Computes intersection over union (dice) of between a set of bounding boxes
    and a set of anchor boxes.

    # Arguments
        - bboxes: numpy array of shape [number_gt_boxes, 5] with bboxes
                  encoded as [cx, cy, w, h, class]
        - anchors: Keras tensor of shape [number_boxes, 4]
                   encoded as [cx, cy, w, h]
    # Returns
        - IoU: Keras tensor of shape [number_boxes, number_gt_boxes] with IoU of
               each anchor with the given box.
    '''
    bboxes_upper_left = bboxes[:,:2] - 0.5 * bboxes[:,2:4]
    bboxes_bottom_right = bboxes_upper_left + bboxes[:,2:4]
    bboxes_upper_left = K.reshape(bboxes_upper_left, (-1,1,2))
    bboxes_bottom_right = K.reshape(bboxes_bottom_right, (-1,1,2))

    anchors_upper_left = anchors[:,:2] - 0.5 * anchors[:,2:]
    anchors_bottom_right = anchors_upper_left + anchors[:,2:]
    anchors_upper_left = K.reshape(anchors_upper_left, (1,-1,2))
    anchors_bottom_right = K.reshape(anchors_bottom_right, (1,-1,2))

    # compute intersection
    inter_upleft = K.maximum(anchors_upper_left, bboxes_upper_left)
    inter_botright = K.minimum(anchors_bottom_right, bboxes_bottom_right)
    inter_wh = inter_botright - inter_upleft
    inter_wh = K.maximum(inter_wh, 0)
    inter = inter_wh[:,:, 0] * inter_wh[:,:, 1]
    # compute union
    area_bboxes = K.reshape(bboxes[:,2]*bboxes[:,3], (-1,1))
    area_anchors = K.reshape(anchors[:,2]*anchors[:,3], (1,-1))
    union = area_bboxes + area_anchors - inter
    # compute iou
    iou = inter / union
    return iou

def _argmax(tensor):
    '''
    # Returns
        - tuple containing the index of the maximum in the given 2D tensor
    '''
    shape = K.shape(tensor)
    assert K.ndim(tensor) == 2, "Works only for 2D tensors, this has " + str(K.ndim(shape)) + " instead."
    flat = K.reshape(tensor, (-1,))
    max_index = K.cast(K.argmax(flat), 'int32')
    return (tf.divide(max_index, shape[1]), tf.mod(max_index, shape[1]))

def _assign_boxes(anchors, gt_boxes, overlap_threshold, num_classes):
    '''
    For each anchor find a bounding box with the highest overlap
    Output of this step is an array shape:
    [number_boxes, 4 (= gt loc) + 1 (= indicator) + num_classes (= one hot encoded)]

    Currently only implemented for tf backend

    From the papers
    ----------------------------
    Matching strategy
    During training we need to determine which default boxes corre-
    spond to a ground truth detection and train the network accordingly. For each ground
    truth box we are selecting from default boxes that vary over location, aspect ratio, and
    scale. We begin by matching each ground truth box to the default box with the best
    jaccard overlap (as in MultiBox [7]). Unlike MultiBox, we then match default boxes to
    any ground truth with jaccard overlap higher than a threshold (0.5). This simplifies the
    learning problem, allowing the network to predict high scores for multiple overlapping
    default boxes rather than requiring it to pick only the one with maximum overlap.
    ----------------------------

    # Arguments
        - num_classes: positive int. Number of classes including the background.
    '''
    num_anchors = K.shape(anchors)[0]
    num_gt_boxes = K.shape(gt_boxes)[0]

    # Compute IoU matrix
    ious = _iou(gt_boxes, anchors)
    # Initialize assignment array
    result = tf.zeros(shape=(num_anchors, 5+num_classes), dtype='float32')
    # Initialize array for marking of assigned GT boxes
    gt_assignment = tf.zeros(shape=(num_gt_boxes,), dtype='int32')

    def cond(ious, *args):
        'max(ious) > overlap_threshold'
        return K.greater_equal(tf.reduce_max(ious), overlap_threshold)

    def body(ious, result, gt_assignment):
        max_iou_index = _argmax(ious)
        bbox_index = K.cast(max_iou_index[0], 'int32')
        anchor_index = K.cast(max_iou_index[1], 'int32')

        # Assign anchor to the bbox with the highest overlap
        update = tf.scatter_nd(indices=[[anchor_index]],
                            updates=[[gt_boxes[bbox_index, i] for i in xrange(4)]
                                     + [1.0]
                                     + [tf.where(K.equal(gt_boxes[bbox_index,4], i),
                                                 1.0, 0.0)
                                        for i in xrange(num_classes)]],
                            shape=K.shape(result))
        result = result + update

        # Update IoU matrix to mark this anchor as assigned
        update = 1.0 + K.expand_dims(
                        tf.scatter_nd(indices=[[anchor_index]],
                                      updates=[-1.0],
                                      shape=(num_anchors,)),
                        axis=0) # vector of all 1s and 0 in place of matched anchor box
        ious = ious * update

        # Update gt_assignment
        gt_assignment = gt_assignment + tf.scatter_nd(indices=[[bbox_index]],
                                                      updates=[1],
                                                      shape=(num_gt_boxes,))
        return ious, result, gt_assignment

    ious, result, gt_assignment = tf.while_loop(cond, body, [ious, result, gt_assignment])
    
    def check_all_assigned(result, assignment):
        unassigned = assignment==0
        if np.any(unassigned):
            print 'Some GT boxes were not assigned.'
            print assignment
        return result
    result = tf.py_func(check_all_assigned, [result, gt_assignment], 'float32', False)

    return result


def MultiboxLoss(y_true, y_pred, overlap_threshold=0.5, num_classes=4):
    '''
    Computes loss for the SSD network.

    1. Performs assignment of gt boxes to anchors
    2. Selects hard negatives
    3. Compute SoftL1 loss on locations
    4. Compute cross entropy on confidence

    # Arguments
        - y_true: List of GT boxes for each image in batch. Format:
                  [ np.array([[center_x, center_y, width, height, class],
                              [...]]) ]
        - y_pred: Tensor of predictions of the network. Format:
                  shape = (batch_size, number_boxes, 4 (= loc) + num_classes (= conf) + 4 (= anchors))
    '''



    pass
    #anchors = y_pred[]

def test():
    bboxes = np.array([[10,10,10,10,1],
                       [20,10,10,10,2],
                       [10,20,10,10,3]])
    anchors = np.array([[5,5,10,10],
                        [5,10,10,10],
                        [10,5,10,10],
                        [10,10,8,8],
                        [20,8,10,10],
                        [22,12,10,10]])
    a = K.placeholder(shape=(None, 5))
    b = K.placeholder(shape=(None, 4))
    # c = _iou(a,b)
    # d = tf.reduce_max(c)
    # d = tf.greater_equal(d, 0.5)
    c = _assign_boxes(b,a, 0.1, 4)
    f = K.function([a,b], [c])
    assignment = f([bboxes, anchors])
    #print assignment
    return assignment

test()

#%%
import tensorflow as tf
import keras.backend as K
import numpy as np

c = tf.constant(6)
b = tf.constant(10)
a = tf.mod(b,c)
d = tf.div(b,c)
f = K.function([], [d,a])
f([])
