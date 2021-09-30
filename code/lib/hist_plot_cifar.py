import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as pltlines
import matplotlib.ticker as mtick

import glob

#from .plotters import matplotlib_config

def plot_experiment_hist_one_row(file_dir, row, x=None, mode='max', ls='-',
    log = True, skip_first=False):
    data = _read_csvs(file_dir)
    
    amin = np.amin(data[row,1:,:], axis=0)
    if log:
        T = lambda x: np.log(x)
    else:
        T = lambda x: x

    if mode == 'all':
        for s in range(0, data.shape[2]):
            for i in range(0, data.shape[3]):
                y = data[row,1:,s,i]
                colour = str(np.clip(data[row,0,s,i], 0, 1))
                if x is None:
                    plt.plot(T(y), alpha=0.1, c=colour, linewidth=1, ls=ls)
                else:
                    plt.plot(x, T(y), alpha=0.1, c=colour, linewidth=1, ls=ls)
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

def plot_epochs_individually(out_dir, data, log=False, ylim=1, test_or_train='Test'):
    data = np.nan_to_num(data, nan=ylim, posinf=ylim)

    if test_or_train == 'Test':
        test_glm = data[1,1:,:,:]
        test_rand = data[3,1:,:,:]

        glm_sn = data[1,0,:,:]
        rand_sn = data[3,0,:,:]
    elif test_or_train == 'Train':
        test_glm = data[0,1:,:,:]
        test_rand = data[2,1:,:,:]

        glm_sn = data[0,0,:,:]
        rand_sn = data[2,0,:,:]

    # For every epoch
    for epoch in range(0, test_glm.shape[0]):
        print('    PLOTTING EPOCH ' + str(epoch) + '...')
        test_glm_e = np.clip(test_glm[epoch,:,:], 0, ylim)
        test_rand_e = np.clip(test_rand[epoch,:,:], 0, ylim)

        for seed in range(0, test_glm_e.shape[1]):
            plt.scatter(glm_sn[:, seed], test_glm_e[:, seed], c='b', s = 1, alpha = 0.5)
            plt.scatter(rand_sn[:, seed], test_rand_e[:, seed], c='r', s = 1, alpha = 0.5)
        
        plt.xscale('log')
        logstr = ''
        if log:
            plt.yscale('log')
            logstr = 'log'

        plt.ylim([None,ylim])
        plt.xlabel('Average spectral norm')
        plt.ylabel(test_or_train + ' MSE')

        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.text(0.05, 0.95, 'Epoch = ' + str(epoch), 
            transform=ax.transAxes, fontsize=15)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

        plt.tight_layout()
        plt.savefig(out_dir + test_or_train + '_epoch' + str(epoch) + '_' + logstr + \
            '_' + str(ylim) + '.pdf', bbox_inches='tight')
        plt.close()

def _read_csvs(file_dir):
    """
    Return a (4, NUM_EPOCHS+2, S, NUM_SEED) array, where NUM_EPOCHS is the maximum 
    number of training epochs and S is the number of spectral norms tried.
    NUM_SEED is the number of random trials e.g. 100

    The first axis indexes [train_glm, test_glm, train_rand, test_rand]
    The elements data[:,0,:,:] represent the spectral norms.
    The elements data[:,1:,:,:] represent the errors, with the index running over epochs
    the second-last axis indexes the scale index (which influences the spectral norm)
    The last axis indexes the random seed
    """
    print('READING DATA')
    all_data = None
    for i ,fname in enumerate(glob.glob(file_dir+'*.npy')):
        data = np.load(fname, allow_pickle=True)
        if all_data is None:
            all_data = np.expand_dims(data, axis=len(data.shape))
        else:
            all_data = np.append(all_data, np.expand_dims(data, axis=len(data.shape)),
            len(data.shape))

    print('READ ALL DATA')
    return all_data

if __name__ == '__main__':
    DIR = 'outputs/cifar10/convam/'
    #matplotlib_config()

    data = _read_csvs(DIR)
    
    for epoch in range(data.shape[1]-1):
        test_glm = data[1,epoch+1,:,:]
        test_rand = data[3,epoch+1,:,:]

        test_glm[np.where(test_glm == np.inf)]  = np.nan
        test_rand[np.where(test_rand == np.inf)]  = np.nan

        max_glm = np.nanmax(test_glm, axis=1)
        max_rand = np.nanmax(test_rand, axis=1)

        inds = np.where(np.isnan(test_glm))
        test_glm[inds] = np.take(max_glm, inds[0])

        inds = np.where(np.isnan(test_rand))
        test_rand[inds] = np.take(max_rand, inds[0])

        best_glm = np.nanargmin(np.nanmean(test_glm, axis=1))
        best_rand = np.nanargmin(np.nanmean(test_rand, axis=1))

        mean_glm = np.nanmin(np.nanmean(test_glm, axis=1))
        mean_rand = np.nanmin(np.nanmean(test_rand, axis=1))

        std_glm = np.sqrt(np.nanvar(test_glm, axis=1))[best_glm]
        std_rand = np.sqrt(np.nanvar(test_rand, axis=1))[best_rand]

        num_nans_glm = np.count_nonzero(np.isnan(test_glm[best_glm, :]))
        num_nans_rand = np.count_nonzero(np.isnan(test_rand[best_rand, :]))
    
        print(epoch)
        print('glm')
        print(mean_glm)
        print(np.median(np.nanmean(test_glm, axis=1)))
        print(std_glm)

        print('rand')
        print(mean_rand)
        print(np.median(np.nanmean(test_rand, axis=1)))
        print(std_rand)
    
    for test_or_train in ['Test']:
        plot_epochs_individually(DIR, data, log=True, test_or_train = test_or_train)
        plot_epochs_individually(DIR, data, log=False, test_or_train = test_or_train)
        #plot_epochs_individually(DIR, data, log=True, ylim=1e-2, test_or_train = test_or_train)
        #plot_epochs_individually(DIR, data, log=False, ylim=1e-2, 
        #test_or_train = test_or_train)
        plot_epochs_individually(DIR, data, log=True, ylim=2e-2,
        test_or_train = test_or_train)
        plot_epochs_individually(DIR, data, log=False, ylim=2e-2,
        test_or_train = test_or_train)
        plot_epochs_individually(DIR, data, log=True, ylim=6.5e-3,
        test_or_train = test_or_train)
        plot_epochs_individually(DIR, data, log=False, ylim=6.5e-3,
        test_or_train = test_or_train)
        #plot_epochs_individually(DIR, data, log=True, ylim=7e-3,
        #test_or_train = test_or_train)
        #plot_epochs_individually(DIR, data, log=False, ylim=7e-3,
        #test_or_train = test_or_train)

