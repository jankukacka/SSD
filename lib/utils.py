# ------------------------------------------------------------------------------
#  Single Shot Multibox Detector for Vertebra detection
#  Jan Kukacka, 2/2018
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Implementation of Soft-Non-Maximum Suppression
#  For details see https://arxiv.org/abs/1704.04503.pdf
# ------------------------------------------------------------------------------

import numpy as np

def _gaussian(iou, sigma):
    return np.exp((-iou*iou)/sigma)

def index_in_remaining_to_original(remaining, index):
    '''
    Convert index in a sub-array represented by a boolean vector to an index
    in the original large array.
    # Arguments:
        - remaining: boolean array (list)
        - index: index in the sub-array
    # Returns:
        - index to the original array
    '''
    for i in xrange(len(remaining)):
        if remaining[i]:
            if index == 0:
                return i
            index -= 1

def nms(bboxes, sigma=0.5, cutoff=0.001):
    '''
    '''
    keep = []
    remaining = [True] * bboxes.shape[0]

    upper_left = bboxes[:,:2] - .5*bboxes[:,2:4]
    bottom_right = bboxes[:,:2] + .5*bboxes[:,2:4]
    sizes = bottom_right - upper_left
    areas = sizes[:,0]*sizes[:,1]
    scores = bboxes[:,4]

    while np.any(remaining):
        ## Compute the best bbox and add it to the result
        i = np.argmax(bboxes[remaining,4])
        orig_i = index_in_remaining_to_original(remaining, i)
        keep.append(orig_i)
        remaining[orig_i] = False

        ## Stop if this was the last one
        if not np.any(remaining):
            break

        ## Compute IOUs
        over_top_left = np.maximum(upper_left[i], upper_left[remaining])
        over_bottom_right = np.minimum(bottom_right[i], bottom_right[remaining])
        size = np.maximum(0,over_bottom_right - over_top_left)
        inter = size[:,0] * size[:,1]
        union = areas[orig_i] + areas[remaining] - inter
        iou = inter / (union+np.finfo(np.float32).eps)

        ## Adjust scores
        scores[remaining] = scores[remaining] * _gaussian(iou, sigma)

        ## Prune boxes below threshold
        remaining_scores = scores[remaining]
        remaining_after = remaining[:]
        for i in xrange(len(remaining_scores)):
            if remaining_scores[i] < cutoff:
                orig_i = index_in_remaining_to_original(remaining, i)
                remaining_after[orig_i] = False
        remaining = remaining_after

    return keep

def aggregate_bboxes_ccwh(bboxes, mode):
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[:,:2] = bboxes[:,:2] - .5*bboxes[:,2:]
    new_bboxes[:,2:] = bboxes[:,:2] + .5*bboxes[:,2:]

    bbox = aggregate_bboxes(new_bboxes, mode)
    new_bbox  = np.empty(4)
    new_bbox[:2] = (bbox[2:] + bbox[:2]) / 2
    new_bbox[2:] = bbox[2:] - bbox[:2]
    return new_bbox

def aggregate_bboxes(bboxes, mode='max'):
    '''
    Aggregates several bounding boxes into one.

    # Arguments:
        - bboxes: numpy array of bounding boxes with shape [num_bboxes, 4],
                encoded as [x1, y1, x2, y2]
        - mode: one from the following:
                - 'max': aggreage as hull around all of the bboxes
                - 'mean': agregate mean of all predictions
    '''
    # Conversion from ccwh to xyxy

    bbox = np.empty(4)
    if mode == 'max':
        bbox[:2] = np.min(bboxes[:,:2], axis=0)
        bbox[2:] = np.max(bboxes[:,2:], axis=0)
        return bbox

    if mode == 'mean':
        bbox = np.mean(bboxes, axis=0)
        return bbox



    raise Exception('Invalid option for the parameter mode. Must be "max" or "mean"')
