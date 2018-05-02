import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def configure_latex():
    """
        Configure matplotlib to use latex in a pretty way
    """
    plt.rcParams.update({
        #'font.family': 'serif',
        'font.serif': 'Palatino',
        'font.size': 12,
        'legend.fontsize': 14,
        'legend.labelspacing': 0,
        'text.usetex': True,
        'savefig.dpi': 300})


def simplify_axis(axis):
    """
        Simplify axis
    """
    axis.set_frame_on(False)
    xmin, xmax = axis.get_xaxis().get_view_interval()
    ymin, ymax = axis.get_yaxis().get_view_interval()
    axis.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black',
                               linewidth=1, zorder=100, clip_on=False))
    axis.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black',
                               linewidth=1, zorder=100, clip_on=False))
    axis.get_yaxis().tick_left()
    axis.get_xaxis().tick_bottom()


def save_fig(fig_name, width=9, height=5):
    """
        Save the current figure under filename fig_name
    """
    plt.gcf().set_size_inches((width, height))
    simplify_axis(plt.gca())
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.02)


def plot(fn, color=None):
    df = pd.read_csv(fn)
    before = np.linalg.norm(df[['x0', 'x1']] - df[['g0', 'g1']].values, axis=1)
    after = np.linalg.norm(df[['x_0', 'x_1']] - df[['g0', 'g1']].values, axis=1)
    #n, bins, patches = plt.hist(after - before, bins=64, cumulative=True, density=True, alpha=0.0, histtype='step')
    n, bins = np.histogram(after - before, bins=64, density=True)
    mean = (bins[:-1] * n / n.sum()).sum()
    n = np.cumsum(n)
    #if bins[-1] < 2.0:
    #    bins[-2] = 2.0
    #    bins[-1] = 2.0
    n /= n[-1]
    plt.fill_between((bins[:-1] + bins[1:]) / 2, n, n * 0, alpha=0.5)
    return mean

configure_latex()

#plot('cleaned_human_slippery.csv', color='blue')
#plot('cleaned_learned_slippery.csv', color='orange')
##plt.ylim(0.0, 0.3)
#plt.xlim(-0.8, 2.0)
#plt.show()

def plot_outer(fn1, fn2, xmax=10.0, legend=True):
    m1 = plot(fn1, color='blue')
    m2 = plot(fn2, color='orange')
    plt.ylim(0.0, 1.0)
    plt.xlim(-0.8, xmax)
    plt.xlabel('Increase in distance to goal')
    plt.ylabel('Cumulative distribution')
    plt.vlines(m1, ymin=0.0, ymax=1.0, color='blue', linestyles='dotted')
    plt.vlines(m2, ymin=0.0, ymax=1.0, color='orange', linestyles='dotted')
    if legend:
        plt.legend(['Simple (cdf)', 'Learned (cdf)', 'Simple (mean)', 'Learned (mean)'], loc='upper left', frameon=False)
    simplify_axis(plt.gca())


plot_outer('cleaned_human.csv', 'cleaned_learned.csv', xmax=0.55)
save_fig('non_slippery.pdf', width=4.5, height=4)
plt.show()
plot_outer('cleaned_human_slippery.csv', 'cleaned_learned_slippery.csv', xmax=2.0, legend=False)
save_fig('slippery.pdf', width=4.5, height=4)
plt.show()
