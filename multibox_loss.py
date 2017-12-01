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
        - bboxes: numpy array of shape [batch_size, number_gt_boxes, 5] with bboxes
                  encoded as [cx, cy, w, h, class]
        - anchors: Keras tensor of shape [batch_size, number_boxes, 4]
                   encoded as [cx, cy, w, h]
    # Returns
        - IoU: Keras tensor of shape [batch_size, number_boxes, number_gt_boxes]
               with IoU of each anchor with the given box.
    '''
    bboxes_upper_left = bboxes[:,:,:2] - 0.5 * bboxes[:,:,2:4]
    bboxes_bottom_right = bboxes_upper_left + bboxes[:,:,2:4]
    bboxes_upper_left = K.expand_dims(bboxes_upper_left, axis=-2)
    bboxes_bottom_right = K.expand_dims(bboxes_bottom_right, axis=-2)

    anchors_upper_left = anchors[:,:,:2] - 0.5 * anchors[:,:,2:]
    anchors_bottom_right = anchors_upper_left + anchors[:,:,2:]
    anchors_upper_left = K.expand_dims(anchors_upper_left, axis=-3)
    anchors_bottom_right = K.expand_dims(anchors_bottom_right, axis=-3)

    # compute intersection
    inter_upleft = K.maximum(anchors_upper_left, bboxes_upper_left)
    inter_botright = K.minimum(anchors_bottom_right, bboxes_bottom_right)
    inter_wh = inter_botright - inter_upleft
    inter_wh = K.maximum(inter_wh, 0)
    inter = inter_wh[:,:,:, 0] * inter_wh[:,:,:, 1]
    # compute union
    area_bboxes = K.expand_dims(bboxes[:,:,2]*bboxes[:,:,3], axis=-1)
    area_anchors = K.expand_dims(anchors[:,:,2]*anchors[:,:,3], axis=-2)
    union = area_bboxes + area_anchors - inter
    # compute iou
    iou = inter / (union+K.epsilon())
    return iou

def _argmax(tensor):
    '''
    # Returns
        - tuple containing the indices of the maximas in the last 2 dimensions
          in the given 3D tensor
    '''
    shape = K.shape(tensor)
    flat = K.reshape(tensor, (shape[0], -1,))
    max_index = K.cast(K.argmax(flat), 'int32')
    return (tf.divide(max_index, shape[-1]), tf.mod(max_index, shape[-1]))

def _assign_boxes(anchors, gt_boxes, overlap_threshold, num_classes):
    '''
    For each anchor find a bounding box with the highest overlap
    Output of this step is an array shape:
    [batch_size, number_boxes, 4 (= gt loc) + 1 (= indicator) + num_classes (= one hot encoded)]

    Currently only implemented for tf backend

    # Arguments
        - anchors:
        - gt_boxes:
        - num_classes: positive int. Number of classes including the background.
        - overlap threshold: float in range [0;1]. Minimum threshold to consider
                             an anchor responsible for a GT box
    '''
    batch_size = K.shape(anchors)[0]
    num_anchors = K.shape(anchors)[1]
    num_gt_boxes = K.shape(gt_boxes)[1]

    # Compute IoU matrix
    ious = _iou(gt_boxes, anchors)
    # Initialize assignment array
    # keras backend currently does not support variable sized zeros tensor
    result = tf.zeros(shape=(batch_size, num_anchors, 5+num_classes), dtype='float32')
    # Initialize array for marking of assigned GT boxes
    gt_assignment = tf.zeros(shape=(batch_size, num_gt_boxes,), dtype='int32')

    def cond(ious, *args):
        'max(ious) > overlap_threshold'
        return K.any(K.greater_equal(K.max(ious), overlap_threshold))

    def body(ious, result, gt_assignment):
        max_iou_index = _argmax(ious)
        bbox_index = K.cast(max_iou_index[0], 'int32')
        anchor_index = K.cast(max_iou_index[1], 'int32')

        gtbox_indices = tf.stack((tf.range(batch_size), bbox_index), axis=-1)
        selected_boxes = tf.gather_nd(gt_boxes, gtbox_indices)

        # Assign anchor to the bbox with the highest overlap
        indices = tf.stack((tf.range(batch_size), anchor_index), axis=-1)
        updates = tf.concat((selected_boxes[:,:4],
                             tf.ones(shape=(batch_size, 1)),
                             tf.one_hot(K.cast(selected_boxes[:,4], 'int32'), depth=num_classes)), axis=-1)

        updates = tf.Print(updates, [tf.shape(updates)], message='updates.shape')
        update = tf.scatter_nd(indices=indices,
                               updates=updates,
                               shape=K.shape(result))

        # Mask updates where max overlap is < overlap_threshold
        ious_indices = tf.stack((tf.range(batch_size),
                                 bbox_index,
                                 anchor_index), axis=-1)
        max_ious = tf.gather_nd(ious, ious_indices)
        is_over_threshold = K.reshape(
                                tf.where(tf.greater_equal(max_ious,overlap_threshold),
                                         tf.ones(shape=(batch_size,)),
                                         tf.zeros(shape=(batch_size,))),
                                (-1,1,1))
        result = result + is_over_threshold * update

        # Update IoU matrix to mark this anchor as assigned
        update = 1.0 + K.expand_dims(
                        tf.scatter_nd(indices=indices,
                                      updates=-1.0*tf.ones(shape=(batch_size,)),
                                      shape=(batch_size, num_anchors,)),
                        axis=1) # vector of all 1s and 0 in place of matched anchor box
        ious = ious * update

        # Update gt_assignment
        gt_assignment = gt_assignment + tf.scatter_nd(indices=gtbox_indices,
                                                      updates=tf.ones(shape=(batch_size,), dtype='int32'),
                                                      shape=(batch_size, num_gt_boxes,))
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

def _l1_smooth_loss(y_true, y_pred):
    """Compute L1-smooth loss.
    # Arguments
        y_true: Ground truth bounding boxes,
            tensor of shape (?, num_boxes, 4).
        y_pred: Predicted bounding boxes,
            tensor of shape (?, num_boxes, 4).
    # Returns
        l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).
    # References
        https://arxiv.org/abs/1504.08083
    """
    abs_loss = K.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(K.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    return K.sum(l1_loss, axis=-1)

def _cross_entropy(y_true, y_pred):
    """Computes cross entropy loss
    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits (outputs of a softmax),
            tensor of shape (?, num_boxes, num_classes).
    # Returns
        cross_entropy: Cross entropy, tensor of shape (?, num_boxes).
    """
    y_pred = tf.Print(y_pred, [y_pred, tf.shape(y_pred), y_true, tf.shape(y_true)], message='y_pred, y_pred.shape, y_true, y_true.shape:', summarize=100)
    y_pred = K.maximum(K.minimum(y_pred, 1 - K.epsilon()), K.epsilon())
    cross_entropy = - K.sum(y_true * K.log(y_pred), axis=-1)
    return cross_entropy

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
    anchors = y_pred[:,:,-4:]

    gt = _assign_boxes(anchors=anchors, gt_boxes=y_true,
                       overlap_threshold=overlap_threshold,
                       num_classes=num_classes)

    targets = tf.concat(((gt[:,:,:2] - anchors[:,:,:2]) / anchors[:,:,-2:],
                          tf.log(gt[:,:,2:4]/anchors[:,:,-2:])), axis=-1)
    loc_error = _l1_smooth_loss(y_true=gt[:,:,:4], y_pred=targets)
    conf_error = _cross_entropy(y_true=gt[:,:,5:], y_pred=y_pred[:,:,4:-4])
    #anchors = y_pred[]
    return tf.where(tf.equal(gt[:,:,4], 1.), loc_error, tf.zeros_like(loc_error)), tf.where(tf.equal(gt[:,:,4], 1.), conf_error, tf.zeros_like(conf_error))

def test():
    bboxes = np.array([[[10,10,10,10,1/4],
                        [0,0,0,0,0],
                        [0,0,0,0,0]],
                       [[10,10,10,10,1],
                        [20,10,10,10,2],
                        [10,20,10,10,3]]]) * np.array([[[4]],[[1]]])
    anchors = np.array([[[5,5,10,10],
                         [5,10,10,10],
                         [10,5,10,10],
                         [0,0,0,0],
                         [20,8,10,10],
                         [22,12,10,10]],
                        [[5,5,10,10],
                         [5,10,10,10],
                         [10,5,10,10],
                         [10,10,8,8],
                         [20,8,10,10],
                         [22,12,10,10]]]) * np.array([[[4]],[[1]]])

    # bboxes = np.array([[[10,10,10,10,1],
    #  [20,10,10,10,2],
    #  [10,20,10,10,3]],
    #                    [[10,10,10,10,1],
    #                     [20,10,10,10,2],
    #                     [10,20,10,10,3]]])
    # anchors = np.array([[[5,5,10,10],
    #                      [5,10,10,10],
    #                      [10,5,10,10],
    #                      [10,10,8,8],
    #                      [20,8,10,10],
    #                      [22,12,10,10]],
    #                     [[5,5,10,10],
    #                      [5,10,10,10],
    #                      [10,5,10,10],
    #                      [10,10,8,8],
    #                      [20,8,10,10],
    #                      [22,12,10,10]]])
    a = K.placeholder(shape=(None,None, 5))
    b = K.placeholder(shape=(None,None, 4))
    #c = _iou(a,b)
    # d = tf.reduce_max(c)
    # d = tf.greater_equal(d, 0.5)
    #c = _assign_boxes(b,a, 0.1, 4)
    c,d = MultiboxLoss(a,b,0.1,4)
    f = K.function([a,b], [c,d])
    assignment = f([bboxes, anchors])
    #print assignment
    return assignment

test()

#%%
import tensorflow as tf
import keras.backend as K
import numpy as np

a = tf.placeholder(shape=(None,))
b = tf.constant(10)
c = tf.constant(6)
d = tf.div(b,c)
f = K.function([], [d,a])
f([])
