import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as pltlines

import glob

from .plotters import matplotlib_config

def plot_experiment_hist_one_row(file_dir, row, x=None, colour='b', mode='max',
    log = True, skip_first=False):
    data = _read_csvs(file_dir)
    if len(data.shape) > 3:
        data = data[:,:,:,0]
    amin = np.amin(data[row,:,:], axis=0)
    if log:
        T = lambda x: np.log(x)
    else:
        T = lambda x: x

    if mode == 'all':
        for s in range(0, data.shape[2]):
            y = data[row,:,s]
            if x is None:
                plt.plot(T(y), alpha=0.1, c=colour, linewidth=1)
            else:
                plt.plot(x, T(y), alpha=0.1, c=colour, linewidth=1)
        plt.savefig('outputs/all_plots.pdf', bbox_inches='tight')
        return
    elif mode == 'min':
        f = np.amin
        ls = '-.'
    elif mode == 'max':
        f = np.amax
        ls = '-'
    elif mode == 'mean':
        f = np.mean
        ls = ':'
    
    length = data.shape[1]
    if x is None:
        plt.plot(f(T(amin))*np.ones_like(data[row,:,0])[:int(length/10)], 
                c=colour, linewidth=1, ls=ls)
    else:
        plt.plot(x[0:10], f(T(amin))*np.ones_like(data[row,:,0])[:int(length/10)], 
                c=colour, linewidth=1, ls=ls)

def _read_csvs(file_dir):
    all_data = None
    for i ,fname in enumerate(glob.glob(file_dir+'*.csv')):
        data = np.genfromtxt(fname)
        if all_data is None:
            all_data = np.expand_dims(data, axis=len(data.shape))
        else:
            all_data = np.append(all_data, np.expand_dims(data, axis=len(data.shape)),
            len(data.shape))

    return all_data

if __name__ == '__main__':
    DIR = 'outputs/smooth/'

    log = True
    matplotlib_config()

    plt.figure(figsize=(10,7))

    plot_experiment_hist_one_row(DIR, row=1, colour='b', mode='all', log=log)
    plot_experiment_hist_one_row(DIR, row=3, colour='r', mode='all', log=log)
    plot_experiment_hist_one_row(DIR, row=5, colour='g', mode='all', log=log)

    plot_experiment_hist_one_row(DIR, row=1, colour='b', mode='min', log=log)
    plot_experiment_hist_one_row(DIR, row=3, colour='r', mode='min', log=log)
    plot_experiment_hist_one_row(DIR, row=5, colour='g', mode='min', log=log)

    plot_experiment_hist_one_row(DIR, row=1, colour='b', mode='max', log=log)
    plot_experiment_hist_one_row(DIR, row=3, colour='r', mode='max', log=log)
    plot_experiment_hist_one_row(DIR, row=5, colour='g', mode='max', log=log)

    plot_experiment_hist_one_row(DIR, row=1, colour='b', mode='mean', log=log)
    plot_experiment_hist_one_row(DIR, row=3, colour='r', mode='mean', log=log)
    plot_experiment_hist_one_row(DIR, row=5, colour='g', mode='mean', log=log)
    
    # Manually construct legend
    blue_patch = mpatches.Patch(color='blue', label='Informed')
    red_patch = mpatches.Patch(color='red', label='Naive')
    green_patch = mpatches.Patch(color='green', label='Random')

    linestyles = ['-.', ':', '-']
    labels = ['Min', 'Mean', 'Max']

    lines = [pltlines.Line2D([0], [0], color='k', linewidth=3, 
        linestyle=linestyles[i], label=labels[i]) \
                for i in range(len(linestyles))]

    plt.legend(handles=[blue_patch, red_patch, green_patch] + lines,
            handlelength=4, fontsize=30)

    plt.xlabel('Training epoch')
    plt.ylabel('$\log$ MSE')
    plt.ylim([None,np.log(0.4)])
    plt.tight_layout()
    plt.savefig(DIR + 'all_plots.pdf', bbox_inches='tight')

