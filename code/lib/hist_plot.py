import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob

from .plotters import matplotlib_config

def plot_experiment_hist_one_row(file_dir, row, x=None, colour='b', mode='max'):
    data = _read_csvs(file_dir)
    amin = np.amin(data[row,:,:], axis=0)

    if mode == 'all':
        for s in range(0, data.shape[2]):
            y = data[row,:,s]
            if x is None:
                plt.plot(np.log(y), alpha=0.3, c=colour, linewidth=1)
            else:
                plt.plot(x, np.log(y), alpha=0.3, c=colour, linewidth=1)
        plt.savefig('outputs/all_plots.pdf')
        return
    elif mode == 'min':
        f = np.amin
        ls = '-.'
    elif mode == 'max':
        f = np.amax
        ls = '-'
    elif mode == 'mean':
        f = np.mean
        ls = '--'

    if x is None:
        plt.plot(f(np.log(amin))*np.ones_like(data[row,:,0]), c=colour, 
                linewidth=3, ls=ls)
    else:
        plt.plot(x, f(np.log(amin))*np.ones_like(data[row,:,0]), c=colour, 
                linewidth=3, ls=ls)

def _read_csvs(file_dir):
    all_data = None
    for i ,fname in enumerate(glob.glob(file_dir+'*.csv')):
        data = np.genfromtxt(fname)
        if all_data is None:
            all_data = np.expand_dims(data, axis=2)
        else:
            all_data = np.append(all_data, np.expand_dims(data, axis=2), 2)

    return all_data

if __name__ == '__main__':
    #DIR = 'outputs/noise_var_05/'
    DIR = 'outputs/'
    matplotlib_config()

    plot_experiment_hist_one_row(DIR, row=1, colour='b', mode='all')
    plot_experiment_hist_one_row(DIR, row=3, colour='r', mode='all')
    plot_experiment_hist_one_row(DIR, row=5, colour='g', mode='all')

    plot_experiment_hist_one_row(DIR, row=1, colour='b', mode='min')
    plot_experiment_hist_one_row(DIR, row=3, colour='r', mode='min')
    plot_experiment_hist_one_row(DIR, row=5, colour='g', mode='min')

    plot_experiment_hist_one_row(DIR, row=1, colour='b', mode='max')
    plot_experiment_hist_one_row(DIR, row=3, colour='r', mode='max')
    plot_experiment_hist_one_row(DIR, row=5, colour='g', mode='max')

    plot_experiment_hist_one_row(DIR, row=1, colour='b', mode='mean')
    plot_experiment_hist_one_row(DIR, row=3, colour='r', mode='mean')
    plot_experiment_hist_one_row(DIR, row=5, colour='g', mode='mean')
    
    # Manually construct legend
    blue_patch = mpatches.Patch(color='blue', label='Informed GLM')
    red_patch = mpatches.Patch(color='red', label='Naive GLM')
    green_patch = mpatches.Patch(color='green', label='Naive GLM')

    linestyles = ['-.', '-', '--']
    lines = [Line2D([0], [0], color='k', linewidth=3, linestyle=ls) \
    for ls in linestyles]
    labels = ['Min', 'Max', 'Mean']

    plt.legend(handles=[blue_patch, red_patch, green_patch,])
    plt.legend(lines, labels)

    plt.xlabel('Training epoch')
    plt.ylabel('$\log$ MSE')
    plt.tight_layout()
    plt.savefig('outputs/all_plots.pdf')

