# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Implementation of Multibox loss
# ------------------------------------------------------------------------------

import keras.backend as K
import tensorflow as tf

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
    #y_pred = tf.Print(y_pred, [y_pred, tf.shape(y_pred), y_true, tf.shape(y_true)], message='y_pred, y_pred.shape, y_true, y_true.shape:', summarize=100)
    y_pred = K.maximum(K.minimum(y_pred, 1 - K.epsilon()), K.epsilon())
    cross_entropy = - K.sum(y_true * K.log(y_pred), axis=-1)
    return cross_entropy

def MultiboxLoss(y_true, y_pred, overlap_threshold=0.5, num_classes=4, alpha=1):
    '''
    Computes loss for the SSD network.

    1. Compute SoftL1 loss on locations
    2. Compute cross entropy on confidence

    # Arguments
        - y_true: List of desired predictions of the format:
                  (batch_size, number_boxes, 4 (= desired loc) + num_classes)
        - y_pred: Tensor of predictions of the network. Format:
                  shape = (batch_size, number_boxes, 4 (= loc) + num_classes (= conf))
        - num_classes: positive int. Number of predicted classes incl. background.
        - overlap_threshold: float in range [0;1]. Minimum threshold to consider
                             an anchor to be responsible for a GT box.
        - alpha: positive float. Multiplier of the classification error part of
                 the loss. See Returns for details.

    # Returns
        - loss: 0D tensor localization error + alpha * classification error
    '''
    loc_error = _l1_smooth_loss(y_true=y_true[:,:,:4], y_pred=y_pred[:,:,:4])
    conf_error = _cross_entropy(y_true=y_true[:,:,4:], y_pred=y_pred[:,:,4:])

    ## Mask out loss of invalid anchors (have sum of classes == 0)
    loc_error = tf.where(tf.equal(K.sum(y_true[:,:,4:], axis=-1), 1.), loc_error, tf.zeros_like(loc_error))
    conf_error = tf.where(tf.equal(K.sum(y_true[:,:,4:], axis=-1), 1.), conf_error, tf.zeros_like(conf_error))

    ## Mask out localization loss of negative samples
    loc_error = tf.where(tf.equal(y_true[:,:,4], 1.), tf.zeros_like(loc_error), loc_error)

    loss = K.sum(loc_error+alpha*conf_error, axis=-1, keepdims=True)
    normalizer = K.expand_dims(K.sum(y_true[:,:,4:], axis=(1,2)), axis=-1)

    loss = (loss / (normalizer + K.epsilon()))
    return loss
