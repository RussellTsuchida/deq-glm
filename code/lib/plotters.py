import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import gc

def matplotlib_gc():
    plt.cla()
    plt.clf()
    plt.close('all')
    gc.collect()

def matplotlib_config():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['axes.labelsize'] = 30
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 20
    matplotlib.rcParams['ytick.labelsize'] = 20
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

def plot_1d_sequence_data(x, y, x_plot, y_plot, fname, xlim=None, ylim=None):
    alpha = 0.5
    plt.figure()
    pmax = min(5, x.shape[0])

    for p in range(pmax):
        plt.plot(x[p,:], y[p,:],
                label='train_input'+str(p), alpha=alpha)
    plt.plot(x_plot[0,:], y_plot[0,:], lw=2, c='k', label='gt')
    if not (xlim is None):
        plt.xlim(xlim)
    if not (ylim is None):
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

