from plottable import ColumnDefinition, Table
from plottable.plots import bar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import cividis
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as color
import seaborn as sns

from computage.utils.data_utils import cond2class
from scipy.stats import norm
import pandas as pd

def plot_class_bench(results, figsize=(12.5, 7), firstcolwidth=4.1):
    """
        Docstring ...
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

    #############################
    ### PREPARE COLUMN FORMAT ###
    #############################
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
                        width=1.0,
                        plot_fn=bar,
                        textprops={"ha": "center"},
                        plot_kw={
                            "cmap": cmap,
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
                        width=1.2,
                        plot_fn=bar,
                        border="left",
                        textprops={"ha": "center"},
                        plot_kw={
                            "cmap": cividis,
                            "plot_bg_bar": True,
                            "annotate": True,
                            "height": 0.9,
                            "lw": 0.5,
                            "formatter": form(classcounts['Total']) 
                            },
                        ),
            ColumnDefinition(
                name="Model",
                textprops={"ha": "right", "weight": "bold"},
                width=firstcolwidth,
            )
    ]

    ### PREPARE COLUMN FORMAT ###   


    fig, ax = plt.subplots(figsize=figsize)

    table = Table(
        vals.head(10),
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        odd_row_color="#ffffff", 
        even_row_color="#f0f0f0",
        ax=ax,
        textprops={"fontsize": 14},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
    )

    return ax


def plot_mae(result, figsize=(5.5, 3)):
    """
        Docstring
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    color_iters = sns.color_palette('muted') 

    z = norm.ppf(0.95) #95% CI
    result['CI'] = result['MAE_SE'] * z
    result['CI']

    axes.grid(alpha=0.3, zorder=0)
    sns.barplot(data=result, x='index', y='MAE', orient='v', ax=axes, palette=color_iters, zorder=100, )
    axes.errorbar(data=result, x='index', y='MAE', yerr='CI', 
                ls='', lw=1.5, color='black', zorder=200)
    axes.set_xlabel('')
    axes.set_ylabel(f'Mean Absolute Error, years')
    axes.set_title('Chronological age prediction accuracy')
    ytickmax = round(result['MAE'].max())
    axes.set_yticks(range(0, ytickmax + 5, 5))
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')
    axes.set_ylim([0, ytickmax + 3])

    return axes
