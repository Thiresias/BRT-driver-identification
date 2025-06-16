import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

def compute_video_level_AUC(label_list, pred_list, name_list, disp=False):
    df = pd.DataFrame({'label': label_list, 'pred': pred_list, 'videoID': name_list})
    df_grouped = df.groupby(['label', 'videoID']).pred.mean().reset_index()
    # print(df.head())
    # Rename the 'pred' column to 'average_prediction'
    df_grouped.rename(columns={'pred': 'average_prediction'}, inplace=True)
    df_grouped.to_csv('res.csv')
    # Compute roc curve and then, auc
    fpr, tpr, thresholds = roc_curve(df_grouped.label, df_grouped.average_prediction)
    computed_AUC = auc(fpr, tpr)

    if disp:
        df_grouped.style

    return computed_AUC


def pca_analysis(hidden, name_list):
    df = pd.DataFrame({'hidden': hidden, 'videoID': name_list})
    df_grouped = df.groupby(['videoID']).hidden.mean().reset_index()

def calculate_fnr_fpr(pred_list, label_list):
    # Initialize counters
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    true_negatives = 0

    # Iterate through the lists and count the instances of TP, TN, FP, FN
    for score, label in zip(pred_list, label_list):
        pred = 1 if score>0.5 else 0
        if label == 1 and pred == 0:
            false_negatives += 1
        elif label == 0 and pred == 1:
            false_positives += 1
        elif label == 1 and pred == 1:
            true_positives += 1
        elif label == 0 and pred == 0:
            true_negatives += 1

    # Calculate the ratios
    fnr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

    return fnr, fpr
