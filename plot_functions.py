import pandas.core.common as com
from pandas.core.index import Index
from pandas.plotting import scatter_matrix

import itertools
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import confusion_matrix

# plt.style.use('cms')
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (4, 3),
    "lines.markersize": 3,
    "errorbar.capsize": 2,
})

def plot_signal_background(data1, data2, column,
                        channel, sig_sample="",
                        bins=100, **kwargs):

    ## THIS FUNCTION IS FOR PLOTTING SIGNAL VS
    ## BACKGROUND FOR A SPECIFIC VARIABLE/COLUMN


    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.5

    df1 = data1[column]
    df2 = data2[column]

    fig, ax = plt.subplots()
    df1=df1.sample(2000, random_state=1234)
    df2=df2.sample(2000, random_state=1234)
    low = max(min(df1.min(), df2.min()),-5)
    high = max(df1.max(), df2.max())

    ax.hist(df1, bins=bins, range=(low,high), **kwargs)
    ax.hist(df2, bins=bins, range=(low,high), **kwargs)

    # ax.set_yscale('log')

    fig.savefig('{}_{}_{}_{}_vs_{}.pdf'.format(column, channel, sig_sample, data1.multi_class.iloc[0], data2.multi_class.iloc[0]))
    print('Signal/Background plot of {} saved'.format(column))


    return None


def plot_roc_cutbased(data1, data2, column,
        channel, sig_sample):

    ## THIS FUNCTION PLOTS THE ROC CURVE FOR A
    ## CUT-BASED "CLASSIFIER" LIKE m_sv

    sig = data1[column]
    bkg = data2[column]

    sig_hist, sig_bin_edges = np.histogram(
            sig, bins=200,
            range=(0,400),
            weights=data1['wt']
            )
    bkg_hist, bkg_bin_edges = np.histogram(
            bkg, bins=200,
            range=(0, 400),
            weights=data2['wt']
            )

    sig_df = pd.DataFrame(
            data={
                'signal':sig_hist,
                'bin_edges':sig_bin_edges[:-1]
                })
    bkg_df = pd.DataFrame(
            data={
                'background':bkg_hist,
                'bin_edges':bkg_bin_edges[:-1]
                })

    df = pd.merge(sig_df, bkg_df, on='bin_edges')
    print(df)

    for col in ['signal', 'background']:
        df[col] = df[col] / df[col].sum()
    print(df)

    df.sort_values(by='signal', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    sig_bin_max = df['bin_edges'][0]

    df['dist'] = np.abs(sig_bin_max - df['bin_edges'])
    df.sort_values(by='dist', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df)

    df_cum = df.cumsum(axis=0)
    print(df_cum)

    df_zero = pd.DataFrame({'bin_edges':np.nan,'signal':0.0, 'background':0.0}, index=[0])
    df_cum = pd.concat([df_zero, df_cum])

    fig, ax = plt.subplots()
    ax.plot(df_cum['background'], df_cum['signal'])

    fig.savefig('roc_{}_{}_{}.pdf'.format(column, channel, sig_sample))
    print('ROC curve of {} saved'.format(column))


    return None




def plot_roc_curve(fpr, tpr, auc, figname):

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.grid()
    ax.text(0.6, 0.3, 'ROC AUC Score: {0:.4f}'.format(auc),
            bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    fig.savefig(figname)
    print('ROC curve saved as {}'.format(figname))

    return None


def plot_scatter_matrix(X, figname):
    ## THIS FUNCTION CURRENTLY DOESNT DO WHAT IT SHOULD
    ##

    # need to resample DataFrame
    df = X.ix[random.sample(X.index, 100)]
    # df =

    plt.figure()
    sm = scatter_matrix(df, figsize=(20,20), alpha=0.4, s=60, c=['y','r'])
    plt.savefig(figname)
    print('Scatter matrix saved as {}'.format(figname))

    return None


def plot_confusion_matrix(y_test, y_pred, w_test, classes,
                    figname, normalise_by_col=False, normalise_by_row=False,
                    cmap=plt.cm.Blues):

    cm = confusion_matrix(y_test, y_pred, sample_weight=w_test)
    if normalise_by_col:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalised efficiency confusion matrix')
    if normalise_by_row:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        print('Normalised purity confusion matrix')
    else:
        print('Non-normalised confusion matrix')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='w' if cm[i, j] > thresh else 'k')

    plt.tight_layout(pad=1.4)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(figname)
    print('Confusion matrix saved as {}'.format(figname))

    return None


def plot_features(booster, imp_type, figname):

    fig = plt.figure(figsize=(12,7))
    axes = fig.add_subplot(111)

    if imp_type == 'weight':
        xgb.plot_importance(booster, ax=axes, height=0.2, xlim=None, ylim=None,
                title='', xlabel='F score', ylabel='Features', importance_type='weight')
    elif imp_type == 'gain':
        xgb.plot_importance(booster, ax=axes, height=0.2, xlim=None, ylim=None,
                title='', xlabel='F score', ylabel='Features', importance_type='gain')

    fig.savefig(figname)
    print('Feature importance saved as {}'.format(figname))

    return None


def plot_correlation_matrix(data, figname, **kwds):

    corrmat = data.corr(**kwds)

    fig, ax1 = plt.subplots(ncols=1, figsize=(6,3))

    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)

    ax1.set_title("")

    labels = corrmat.columns.values
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, minor=False)

    fig.tight_layout()
    fig.savefig(figname)
    print('Correlation matrix saved as {}'.format(figname))


def plot_output(booster, train, test, y_train, y_test, figname, bins=20, **kwds):

    decisions = []
    for X,y_true in ((train, y_train), (test, y_test)):
        ## needs to be fixed here!!
        d1 = booster.predict(X)[y_true>0.5]
        d2 = booster.predict(X)[y_true<0.5]
        decisions += [d1, d2]
        # print d1, d2
    # print decisions

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    fig, ax = plt.subplots()
    ax.hist(decisions[0], color='r', alpha=0.5, range=low_high, bins=bins,
            histtype='stepfilled', normed=True, label='S(train)')
    ax.hist(decisions[1], color='b', alpha=0.5, range=low_high, bins=bins,
            histtype='stepfilled', normed=True, label='B(train)')

    hist, bins = np.histogram(decisions[2], range=low_high, bins=bins, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    ax.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    ax.set_xlabel("BDT output")
    # ax.set_yscale('log', nonposy='clip')
    ax.set_ylabel("Arbitrary units")
    ax.legend(loc='best')

    fig.savefig(figname)
    print('BDT score saved as {}'.format(figname))

    return None



## FOR GBC
def compare_train_test(clf, X_train, y_train, X_test, y_test, figname, bins=30):
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X.as_matrix()[y>0.5]).ravel()
        d2 = clf.decision_function(X.as_matrix()[y<0.5]).ravel()
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)

    fig, ax = plt.subplots()
    ax.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    ax.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    ax.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    ax.set_xlabel("BDT output")
    ax.set_ylabel("Arbitrary units")
    ax.legend(loc='best')

    fig.savefig(figname)
    print('BDT score saved as {}'.format(figname))

    return None

def plot_learning_curve(model, metric, figname):
    # metric can be mlogloss or merror
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0'][metric])
    x_axis = list(range(0, epochs))

    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0'][metric], label='Train')
    ax.plot(x_axis, results['validation_1'][metric], label='Test')
    ax.legend()
    plt.ylabel(metric)
    plt.title('XGBoost {}'.format(metric))


    fig.savefig(figname)
    print('Learning curve plotted as {}'.format(figname))

    return None
