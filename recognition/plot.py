import os
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

parser     = argparse.ArgumentParser()
parser.add_argument('--csv_hist', default="train.csv", help='Name of training log')
parser.add_argument('--title', help='Plot title', type=str, default="")
parser = parser.parse_args()

def plotloss(csv_hist, title=""):
    '''
    Args
        csv_train: name of the csv file
        csv_val: name of the csv file
    Returns
        graph_loss: trend of loss values over epoch
    '''
    # Bring in the csv file
    hist   = pd.read_csv(csv_hist)
    hist.drop_duplicates(subset="epoch", keep='last', inplace=True)

    
    # Initiation
    epoch           = hist["epoch"]
    tr_loss         = hist["train_loss"]
    val_loss        = hist["test_loss"]

    fig, ax1        = plt.subplots(figsize=(8, 6))

    # Label and color the axes
    ax1.set_xlabel('epoch', fontsize=16)
    ax1.set_ylabel('loss', fontsize=16, color='black')

    # Plot valid/train losses
    ax1.plot(epoch, tr_loss, linewidth=2,
             ls='--', color='#c92508', label='Train loss')
    ax1.plot(epoch, val_loss, linewidth=2,
             ls='--', color='#2348ff', label='Val loss')
    
    for label in ax1.get_xticklabels():
        label.set_size(12)

    # Modification of the overall graph
    fig.legend(ncol=4, loc=9, fontsize=12)
    plt.xlim(xmin=0)
    plt.title(title, weight="bold")
    plt.grid(True, axis='y')

if __name__ == '__main__':
    plt.show(plotloss(parser.csv_hist, parser.title))
