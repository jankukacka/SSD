# --------------------------------------------------------
# Anchor assignment
# Match the set of anchors with given bounding boxes
# Copyright (c) 2017 Jan Kukacka
# --------------------------------------------------------

import numpy as np

def _iou(gt_boxes, anchors):
    '''
    Computes intersection over union (dice) of between a set of bounding boxes
    and a set of anchor boxes.

    # Arguments
        - gt_boxes: numpy array of shape [batch_size, number_gt_boxes, 5] with
                  ground truth bboxes encoded as [cx, cy, w, h, class]
        - anchors: numpy array of shape [batch_size, number_anchors, 4]
                   encoded as [cx, cy, w, h]
    # Returns
        - IoU: numpy array of shape [batch_size, number_anchors, number_gt_boxes]
               with IoU of each anchor with the given box.
    '''
    bboxes_upper_left = gt_boxes[:,:,:2] - 0.5 * gt_boxes[:,:,2:4]
    bboxes_bottom_right = bboxes_upper_left + gt_boxes[:,:,2:4]
    bboxes_upper_left = np.expand_dims(bboxes_upper_left, axis=-2)
    bboxes_bottom_right = np.expand_dims(bboxes_bottom_right, axis=-2)

    anchors_upper_left = anchors[:,:,:2] - 0.5 * anchors[:,:,2:]
    anchors_bottom_right = anchors_upper_left + anchors[:,:,2:]
    anchors_upper_left = np.expand_dims(anchors_upper_left, axis=-3)
    anchors_bottom_right = np.expand_dims(anchors_bottom_right, axis=-3)

    # compute intersection
    inter_upleft = np.maximum(anchors_upper_left, bboxes_upper_left)
    inter_botright = np.minimum(anchors_bottom_right, bboxes_bottom_right)
    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:,:,:, 0] * inter_wh[:,:,:, 1]
    # compute union
    area_bboxes = np.expand_dims(gt_boxes[:,:,2]*gt_boxes[:,:,3], axis=-1)
    area_anchors = np.expand_dims(anchors[:,:,2]*anchors[:,:,3], axis=-2)
    union = area_bboxes + area_anchors - inter
    # compute iou
    iou = inter / (union+np.finfo(np.float32).eps)
    return iou

def _assign_boxes(gt_boxes, anchors, num_classes, overlap_threshold):
    '''
    For each anchor find a bounding box with the highest overlap

    # Arguments
        - gt_boxes:
        - anchors:
        - num_classes: positive int. Number of classes including the background.
        - overlap threshold: float in range [0;1]. Minimum threshold to consider
                             an anchor responsible for a GT box

    # Returns
        - numpy array with shape: [batch_size,
                                   number_boxes,
                                   4 (= gt loc) + 1 (= indicator) + num_classes (= one hot encoded)]
    '''
    batch_size = anchors.shape[0]
    num_anchors = anchors.shape[1]
    num_gt_boxes = gt_boxes.shape[1]

    ## How many positions to leave for indicator
    indicator_width = 0

    # Compute IoU matrix
    ious = _iou(gt_boxes, anchors)
    # Initialize assignment array
    result = np.zeros(shape=(batch_size, num_anchors, 4+indicator_width+num_classes), dtype='float32')
    # assign all as background
    result[:,:,4] = 1

    while(np.max(ious) >= overlap_threshold):
        max_iou_index = np.unravel_index(ious.argmax(), ious.shape)
        batch_index = max_iou_index[0]
        bbox_index = max_iou_index[1]
        anchor_index = max_iou_index[2]

        result[batch_index, anchor_index,:4] = gt_boxes[batch_index, bbox_index,:4]
        gt_class = gt_boxes[batch_index, bbox_index,4]
        result[batch_index, anchor_index,4+indicator_width+int(gt_class)] = 1 # one hot encoding
        result[batch_index, anchor_index,4] = 0 # remove flag from bkg class

        ## Set indicator to 1 for marked anchors
        if indicator_width > 0:
            result[batch_index, anchor_index,4] = 1

        ## Assign no more anchors to this gt box
        #ious[batch_index,bbox_index,:] = 0
        ## Assign this anchor to no more gt boxes
        ious[batch_index,:,anchor_index] = 0
    return result

def _compute_offsets(targets, anchors):
    '''
    For given targets and anchors in the [center_x, center_y, width, height]
    format computes offsets that the network should predict.

    # Arguments
        - targets: numpy array of shape (batch_size, num_anchors, 4 (ccwh))
                   or (num_anchors, 4 (ccwh)).
        - anchors: numpy array of shape (batch_size, num_anchors, 4 (ccwh))
                   or (num_anchors, 4 (ccwh)).
    # Returns
        - offsets: numpy array of the same shape as were the inputs.
    '''
    result = np.zeros_like(targets)
    ## variant for 3D input
    if len(targets.shape) == 3:
        result[:,:,:2] = (targets[:,:,:2] - anchors[:,:,:2]) / anchors[:,:,-2:]
        result[:,:,2:] = np.log(targets[:,:,2:]/anchors[:,:,-2:])
    ## variant for 2D input
    elif len(targets.shape) == 2:
        result[:,:2] = (targets[:,:2] - anchors[:,:2]) / anchors[:,-2:]
        result[:,2:] = np.log(targets[:,2:]/anchors[:,-2:])
    else:
        raise Exception('Invalid data dimension. Supports 3D (batch,anchor,loc) or 2D (anchor, loc)')
    result = np.where(np.isfinite(result), result, np.zeros_like(result))
    return result

def _mark_boundary_boxes(anchors, assignment, input_shape):
    '''
    Finds bounding boxes which cross the image boundary and removes their
    assignment to any class (also as background classes)

    # Arguments
        - anchors: numpy array containing anchor bounding boxes, of shape
                (batch_size, num_anchors, 4 (=loc)).
        - assignment: numpy array containing assignment of anchors to classes,
                of shape (batch_size, num_anchors, num_classes), where
                num_classes includes the background class.
        - input_shape: 2-tuple of (width, height) of the input image

    # Returns
        - assignment: updated assignment field, where anchors on the boundary
                do not belong to any class
    '''
    assignment = np.copy(assignment)
    # print 'valid anchors before', np.sum(assignment)
    assignment = assignment * ((anchors[:,:,:2]-.5*anchors[:,:,2:]) > 0)
    assignment = assignment * (anchors[:,:,:2]+.5*anchors[:,:,2:] <= input_shape)
    # print 'valid anchors after', np.sum(assignment)
    return assignment

def _get_hard_negatives(assignment, num_negative):
    assignment = np.copy(assignment)
    for batch_index in xrange(assignment.shape[0]):
        bkg_anchors = np.where(assignment[batch_index,:] == 1)[0]
        sample = np.random.choice(len(bkg_anchors), max(len(bkg_anchors) - num_negative[batch_index],0), replace=False)
        sample = bkg_anchors[sample]
        assignment[batch_index, sample] = 0
    return assignment

def Match(gt, anchors, num_classes, overlap_threshold, input_shape):
    '''
    Match the set of given anchors with given bounding boxes

    # Arguments
        - gt: numpy array of ground truth bounding boxes in the format:
                (batch_size, ) #TODO: complete docstring.
        - anchors: numpy array of anchors in the format:
                (batch_size, num_anchors, 4 (=center_x, center_y, width, height))
        - num_classes: positive int. Number of classes for 1-hot encoding of the
                result. Includin the background class.
        - overlap_threshold: float in range [0;1]. Minimum threshold to consider
                an anchor to be responsible for a GT box.
        - input_shape: tuple of positive ints. Represents the dimensions of the
                network input. Can be (batch_size, width, height) or (width, height)

    # Returns
        - numpy array of targets in the format:
          (batch_size, num_anchors, 4 (= loc) + num_classes)
    '''
    if len(input_shape) == 3:
        input_shape = input_shape[1:]
    if len(input_shape) != 2:
        raise Exception('Unexpected length of input_shape')

    assignment = _assign_boxes(gt, anchors, num_classes, overlap_threshold)
    assignment[:,:,4:] = _mark_boundary_boxes(anchors, assignment[:,:,4:], input_shape)
    assigned = np.sum(assignment[:,:,5:], axis=-1) != 0
    #print assignment[assigned]
    assignment[assigned,:4] = _compute_offsets(assignment[assigned,:4], anchors[assigned])
    assignment[:,:,4] = _get_hard_negatives(assignment[:,:,4], 3*np.sum(assigned, axis=-1))
    return assignment
