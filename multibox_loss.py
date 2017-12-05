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


    Currently only implemented for tf backend

    # Arguments
        - anchors:
        - gt_boxes:
        - num_classes: positive int. Number of classes including the background.
        - overlap threshold: float in range [0;1]. Minimum threshold to consider
                             an anchor responsible for a GT box

    # Returns
        - Tensor with shape: [batch_size,
                              number_boxes,
                              4 (= gt loc) + 1 (= indicator) + num_classes (= one hot encoded)]
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

    # This messes up the graph (output shapes), so we won't use it.
    def check_all_assigned(result, assignment):
        unassigned = assignment==0
        if np.any(unassigned):
            print 'Some GT boxes were not assigned.'
            print assignment
        return result
#    result = tf.py_func(check_all_assigned, [result, gt_assignment], 'float32', False)

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

def _get_hard_negatives(gt, conf, neg_ratio=3):
    '''
    Selects hard negatives and marks them as valid samples by setting their
    indicator to 1. Keeps ratio of negatives / positives at most at neg_ratio.
    Negatives are selected by their confidence about being background class.

    # Arguments
        - gt: Tensor of shape [batch_size,
                               num_anchors,
                               4 (= gt loc) + 1 (= indicator) + num_classes (= one hot encoded)]
        - conf: Tensor of network predictions for bkg class with shape
                [batch_size, num_anchors, 1]
        - neg_ratio: float. Determines maximum negatives/positives ratio.
                     Default is 3:1 (according to the original paper).
    # Returns
        - updated gt tensor with negative samples' indicators set to 1.
    '''
    batch_size = K.shape(gt)[0]
    num_anchors = K.shape(gt)[1]
    num_classes = K.shape(gt)[2] - 5

    # Compute how many negatives each sample should have as
    # min(unassigned, neg_ratio*num_positives)
    num_unassigned = tf.count_nonzero(K.equal(gt[:,:,4],0), axis=-1, dtype='int32')
    max_negative = neg_ratio * (num_anchors-num_unassigned)
    num_negative = K.minimum(num_unassigned, max_negative)
    num_negative = tf.Print(num_negative, [num_negative], message='num_negative: ', summarize=20)
    max_num_negative = K.max(num_negative)
    max_num_negative = tf.Print(max_num_negative, [max_num_negative], message='max_num_negative: ')

    # Take top_k sorted by the confidence for class 0 multiplied by indicator-1
    # to ensure that no box gets assigned twice.
    # k = max(num_negative)
    _, indices = tf.nn.top_k(conf*(gt[:,:,4]*(-1)), k=max_num_negative)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size),
                                       (batch_size, 1, 1)),
                            (1, max_num_negative, 1))
    indices = tf.concat((batch_indices,
                         tf.expand_dims(indices, axis=-1), # anchor box indices
                         tf.ones_like(batch_indices)*4), axis=-1) # 4 is the indicator index

    # Mask out updates where k > num_negative
    a = tf.tile(tf.expand_dims(tf.range(max_num_negative),axis=0),
                (batch_size, 1))
    b = tf.tile(tf.expand_dims(num_negative, axis=-1),
                (1,max_num_negative))
    updates = tf.where(a<b, tf.ones_like(a, dtype='float32'),
                            tf.zeros_like(a, dtype='float32'))
    updates = tf.Print(updates, [updates, indices], message='updates, indices:', summarize=100)

    # Scatter add 1 to the indicators in the GT tensor
    indicator_update = tf.scatter_nd(indices, updates, shape=K.shape(gt))
    gt = gt + indicator_update
    return gt

def MultiboxLoss(y_true, y_pred, overlap_threshold=0.5, num_classes=4, alpha=1):
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
        - num_classes: positive int. Number of predicted classes incl. background.
        - overlap_threshold: float in range [0;1]. Minimum threshold to consider
                             an anchor to be responsible for a GT box.

    # Returns
        - loss: 0D tensor localization error + alpha * classification error
    '''
    anchors = y_pred[:,:,-4:]

    gt = _assign_boxes(anchors=anchors, gt_boxes=y_true,
                       overlap_threshold=overlap_threshold,
                       num_classes=num_classes)
    gt = _get_hard_negatives(gt=gt, conf=y_pred[:,:,4])

    targets = tf.concat(((gt[:,:,:2] - anchors[:,:,:2]) / anchors[:,:,-2:],
                          tf.log(gt[:,:,2:4]/anchors[:,:,-2:])), axis=-1)
    loc_error = _l1_smooth_loss(y_true=gt[:,:,:4], y_pred=targets)
    conf_error = _cross_entropy(y_true=gt[:,:,5:], y_pred=y_pred[:,:,4:-4])

    # Mask out loss of invalid anchors (have indicator == 0)
    loc_error = tf.where(tf.equal(gt[:,:,4], 1.), loc_error, tf.zeros_like(loc_error))
    conf_error = tf.where(tf.equal(gt[:,:,4], 1.), conf_error, tf.zeros_like(conf_error))

    # Mask out localization loss of negative samples
    loc_error = tf.where(tf.equal(gt[:,:,5], 1.), loc_error, tf.zeros_like(loc_error))

    loss = K.mean(loc_error+alpha*conf_error, axis=-1, keepdims=True)
    #loss = tf.Print(loss, [K.shape(loss)], message='loss.shape', summarize=100)
    return loss

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

    offsets = np.ones_like(anchors)
    preds = np.array([[[0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3]],
                      [[0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3],
                       [0.1,0.5,0.1,0.3]]])

    y_pred = np.concatenate((offsets, preds, anchors),axis=-1)

    a = K.placeholder(shape=(2,3, 5))
    b = K.placeholder(shape=(None, None, 12))
    #c = _iou(a,b)
    # d = tf.reduce_max(c)
    # d = tf.greater_equal(d, 0.5)
    #c = _assign_boxes(b,a, 0.1, 4)
    c = MultiboxLoss(a,b,0.1,4)
    f = K.function([a,b], [c])
    assignment = f([bboxes, y_pred])
    #print assignment
    return assignment, c

#a,c = test()

#print c,K.shape(c)
#%%
# import tensorflow as tf
# import keras.backend as K
# import numpy as np
#
# a = tf.placeholder(shape=(None,))
# b = tf.constant(10)
# c = tf.constant(6)
# d = tf.div(b,c)
# f = K.function([], [d,a])
# f([])
