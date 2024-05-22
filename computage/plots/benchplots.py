from plottable import ColumnDefinition, Table
from plottable.plots import bar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import cividis, viridis
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as color
import seaborn as sns

from computage.utils.data_utils import cond2class
from computage.utils.plot_utils import *
from scipy.stats import norm
import pandas as pd
import numpy as np

def plot_class_bench(results: pd.DataFrame, 
                     figsize=(12.5, 7),
                     classcolwidth=1.0,
                     totalcolwidth=1.1, 
                     firstcolwidth=4.1):
    """
        Plot results of benchmark in a form of formatted table, where
        columns correspond to different classes and rows are different
        models. Entries of the table correspond to number of datasets where
        clocks successfully predicted age acceleration.

        - Can be used for AA2 and AA1 tasks.
        - Accept task boolean results as input.

        returns: matplotlib.pyplot.axes 
    """

    ### prepare data for plotting ###
    conds = [c.split(':')[1] for c in results.columns]
    classes = cond2class(conds)

    df = pd.melt(results.reset_index(), id_vars=['index'])
    df['Condition'] = [c.split(':')[1] for c in df['variable']]
    df['Class'] = cond2class(df['Condition'])
    df = df.drop(['variable', 'Condition'], axis=1)

    sums = df.groupby(['index', 'Class']).sum()
    counts = df.groupby(['index', 'Class']).count()
    sums = pd.pivot(sums.reset_index(), index='index', columns=['Class'], values='value')
    counts = pd.pivot(counts.reset_index(), index='index', columns=['Class'], values='value')
    sums['Total'] = sums.sum(axis=1)
    counts['Total'] = counts.sum(axis=1)
    classcounts = counts.iloc[0]
    vals = sums / counts
    vals.index.name = 'Model'
    vals = vals.sort_values('Total', ascending=False)


    ### PREPARE COLUMN FORMAT ###
    cmap = LinearSegmentedColormap.from_list(
        name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256
    )

    # manual text formatting function like x/y
    def form(base):
        def formatter(x):
            return f'{str(int(round(x * base)))}/{base}'
        return formatter

    col_defs = []
    for col in vals.columns[:-1]:
        base = classcounts[col]
        cldef = ColumnDefinition(
                        col,
                        width=classcolwidth,
                        plot_fn=bar,
                        textprops={"ha": "center", "fontsize":10},
                        plot_kw={
                            "cmap": class_light_cmap_dict[col].reversed(),
                            "plot_bg_bar": True,
                            "annotate": True,
                            "height": 0.9,
                            "lw": 0.5,
                            "formatter": form(base) #apply_formatter()
                            },
                        )
        col_defs.append(cldef)

    col_defs = col_defs + [
        ColumnDefinition(
                        'Total',
                        width=totalcolwidth,
                        plot_fn=bar,
                        border="left",
                        textprops={"ha": "center", "fontsize":10},
                        plot_kw={
                            "cmap": grays_cmap_light,
                            "plot_bg_bar": True,
                            "annotate": True,
                            "height": 0.9,
                            "lw": 0.5,
                            "formatter": form(classcounts['Total']) 
                            },
                        ),
            ColumnDefinition(
                name="Model",
                textprops={"ha": "right", "fontsize":10},
                width=firstcolwidth,
            )
    ]

    ### PLOT ###   
    fig, ax = plt.subplots(figsize=figsize)
    table = Table(
        vals.head(20),
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        odd_row_color="#ffffff", 
        even_row_color="#f0f0f0",
        ax=ax,
        # textprops={"fontsize": 10},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
    )

    return ax


def plot_medae(result: pd.DataFrame, figsize=(5.5, 3), upper_bound=18):
    """
        Plot models median absolute prediction error in a form of barplot.

        - Accept CA_pred_MAE task numerical results as input.

        returns: matplotlib.pyplot.axes 
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    # color_iters = sns.color_palette('muted') 

    color_iters = viridis.reversed()(np.linspace(0, 1, len(result)))
    result['Colors'] = [color.to_hex(c) for c in color_iters.tolist()]

    axes.grid(alpha=0.3, zorder=0)
    sns.barplot(data=result, y='index', x='MAE', orient='h', ax=axes, palette=color_iters, zorder=100, legend=False)
    axes.set_ylabel('')
    axes.set_xlabel(f'Median Absolute $\Delta$, years')
    # axes.set_title('Chronological age prediction accuracy')
    xtickmax = round(result['MAE'].max())
    axes.set_xticks(range(0, xtickmax + 5, 5))
    axes.set_xticklabels(axes.get_xticklabels(), ha='right')
    axes.set_xlim([0, xtickmax + 3])
    axes.axvline(upper_bound, color='grey', ls='--', alpha=0.5)

    for p in axes.patches:
        h = p.get_width() 
        step = 1.5 if h > 0 else -1
        axes.annotate("%.1f" % p.get_width(), 
                        xy=(h - step, p.get_y()+0.4),
                        xytext=(0, 0), 
                        textcoords='offset points', 
                        ha="center", 
                        va="center", 
                        zorder=100,
                        fontweight='bold',
                        color='white',
                        fontsize=8)
    colordict = result[['index', 'Colors']].set_index('index').to_dict()['Colors']
    return axes, colordict


def plot_bias(result: pd.DataFrame, 
              colordict: dict, 
              figsize=(5.5, 3), 
              xlims=[-20, 20]):
    """
        Plot models median prediction error in a form of barplot.

        - Accept CA_pred_bias task numerical results as input.

        returns: matplotlib.pyplot.axes 
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    # color_iters = sns.color_palette('muted') 

    axes.grid(alpha=0.3, zorder=0)
    sns.barplot(data=result, y='index', x='MedE', orient='h', hue='index', legend=False,
                ax=axes, palette=colordict, zorder=100, )
    axes.set_ylabel('')
    axes.set_xlabel(f'Median $\Delta$, years')
    # axes.set_title('Chronological age prediction bias')
    axes.set_xticklabels(axes.get_xticklabels(), ha='right');
    axes.set_xlim(xlims);
    axes.axvline(0, color='grey', ls='--', alpha=0.5)

    for p in axes.patches:
        h = 17 if p.get_width() > 17 else p.get_width()
        h = -17 if p.get_width() < -17 else h
        step = 2.3 if h > 0 else -2.3
        axes.annotate("%.1f" % p.get_width(), 
                        xy=(h + step, p.get_y()+0.37),
                        xytext=(0, 0), 
                        textcoords='offset points', 
                        ha="center", 
                        va="center", 
                        zorder=100,
                        fontweight='bold',
                        fontsize=8)
    return axes
