import numpy as np
import torch


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FN = False judge number
    TN = True judge number
    return: """

    # FN = np.float(np.sum(prediction != groundtruth))
    # TN = np.float(np.sum(prediction == groundtruth))
    FP = np.float(np.sum((prediction != groundtruth) & (prediction != 0)))
    FN = np.float(np.sum((prediction != groundtruth) & (prediction == 0)))
    TP = np.float(np.sum((prediction == groundtruth) & (prediction != 0)))
    TN = np.float(np.sum((prediction == groundtruth) & (prediction == 0)))

    return FP, FN, TP, TN


def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    # FN, TN = numeric_score(prediction, groundtruth)
    # N = FN + TN
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    accuracy = np.divide(TP + TN, TP + TN + FP + FN)
    return accuracy * 100.0


def diceCoeff_avr(prediction, groundtruth):
    """compute dice of  multiple classes"""
    """
    inter_panck = np.float(np.sum(prediction * groundtruth == 1))
    union_panck = np.float(np.sum(prediction == 1) + np.sum(groundtruth == 1))
    inter_nuclei = np.float(np.sum(prediction * groundtruth == 4))
    union_nuclei = np.float(np.sum(prediction == 2) + np.sum(groundtruth == 2))
    inter_lcell = np.float(np.sum(prediction * groundtruth == 9))
    union_lcell = np.float(np.sum(prediction == 3) + np.sum(groundtruth == 3))
    EPS = 1e-5
    dice_panck = (2 * inter_panck + EPS) / (union_panck + EPS)
    dice_nuclei = (2 * inter_nuclei + EPS) / (union_nuclei + EPS)
    dice_lcell = (2 * inter_lcell + EPS) / (union_lcell + EPS)
    dice_avr = np.divide(dice_panck + dice_nuclei + dice_lcell, 3)
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    dice_avr = np.divide(2 * TP, TP + FN +TP +FP)
    return dice_avr


def diceCoeff_panck(prediction, groundtruth):
    inter_panck = np.float(np.sum(prediction * groundtruth == 1))
    union_panck = np.float(np.sum(prediction == 1) + np.sum(groundtruth == 1))
    EPS = 1e-5
    dice_panck = (2 * inter_panck + EPS) / (union_panck + EPS)

    return dice_panck


def diceCoeff_nuclei(prediction, groundtruth):
    inter_nuclei = np.float(np.sum(prediction * groundtruth == 4))
    union_nuclei = np.float(np.sum(prediction == 2) + np.sum(groundtruth == 2))
    EPS = 1e-5
    dice_nuclei = (2 * inter_nuclei + EPS) / (union_nuclei + EPS)

    return dice_nuclei


def diceCoeff_lcell(prediction, groundtruth):
    inter_lcell = np.float(np.sum(prediction * groundtruth == 9))
    union_lcell = np.float(np.sum(prediction == 3) + np.sum(groundtruth == 3))
    EPS = 1e-5
    dice_lcell = (2 * inter_lcell + EPS) / (union_lcell + EPS)

    return dice_lcell


def precision_score(prediction, groundtruth):
    """Getting the precision of the model"""
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    precision = np.divide(TP, TP + FP)
    return precision * 100.0


def recall_score(prediction, groundtruth):
    """Getting the recall of the model"""
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    recall = np.divide(TP, TP + FN)
    return recall * 100.0


def f1_score(prediction, groundtruth):
    """Getting the f1 score of the model"""
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    recall = np.divide(TP, TP + FN)
    precision = np.divide(TP, TP + FP)
    f1 = np.divide(2 * recall * precision, recall + precision)
    return f1 * 100.0


def IOU(prediction, groundtruth):
    """Getting the iou of the model"""
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    iou = np.divide(TP, TP + FN + FP)
    return iou * 100.0