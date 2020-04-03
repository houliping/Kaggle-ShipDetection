import torch
import os
import numpy as np
import time
import warnings

def IOU(pred,target):
    pred=(pred>0).float()
    intersection=(pred*target).sum()
    return intersection/((pred+target).sum()-intersection+1e-8)

def dice(pred, targs):
    pred = (pred>0).float()
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1e-8)

def IoU2(mask1, mask2):
    Inter = np.sum((mask1 >= 0.5) & (mask2 >= 0.5))
    Union = np.sum((mask1 >= 0.5) | (mask2 >= 0.5))
    return Inter / (1e-8 + Union)

def fscore(tp, fn, fp, beta=2.):
    if tp + fn + fp < 1:
        return 1.
    num = (1 + beta ** 2) * tp
    return num / (num + (beta ** 2) * fn + fp)

def confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh=0.5):
    predict_masks = [m for m in predict_mask_seq if np.any(m >= 0.5)]
    truth_masks = [m for m in truth_mask_seq if np.any(m >= 0.5)]

    if len(truth_masks) == 0:
        tp, fn, fp = 0.0, 0.0, float(len(predict_masks))
        return tp, fn, fp

    pred_hits = np.zeros(len(predict_masks), dtype=np.bool) # 0 miss, 1 hit
    truth_hits = np.zeros(len(truth_masks), dtype=np.bool)  # 0 miss, 1 hit

    for p, pred_mask in enumerate(predict_masks):
        for t, truth_mask in enumerate(truth_masks):
            if IOU(pred_mask, truth_mask) > iou_thresh:
                truth_hits[t] = True
                pred_hits[p] = True

    tp = np.sum(pred_hits)
    fn = len(truth_masks) - np.sum(truth_hits)
    fp = len(predict_masks) - tp

    return tp, fn, fp

def mean_fscore(predict_mask_seq, truth_mask_seq,
              iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7,
                              0.75, 0.8, 0.85, 0.9, 0.95], beta=2.):
    """ calculates the average FScore for the predictions in an image over
    the iou_thresholds sets.
    predict_mask_seq: list of masks of the predicted objects in the image
    truth_mask_seq: list of masks of ground-truth objects in the image
    """
    return np.mean(
        [fscore(tp, fn, fp, beta) for (tp, fn, fp) in
            [confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh)
                for iou_thresh in iou_thresholds]])