import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt 

# Helper class for easy object access
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

WIDTH_MAP = {
    'dissertation': 394.35527,
    'ieeeaccess': 505.12177,
    'lncs': 372.0,
}

COLORMAP = ['#e82e82','#009ee3','#984ea3','#ff7f00','#ffbc29','#f781bf','#35cdb4', '#a65628', '#706f6f']
COLORS = objectview({
    'red': COLORMAP[0],
    'blue': COLORMAP[1],
    'violet': COLORMAP[2],
    'orange': COLORMAP[3],
    'yellow': COLORMAP[4],
    'pink': COLORMAP[5],
    'green': COLORMAP[6],
    'brown': COLORMAP[7],
    'gray': COLORMAP[8],
})

major_tick_width = 0.5
minor_tick_width = 0.3

tex_fonts = {
    # Use LaTeX to write all text
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'''
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{bm}''',
    'axes.prop_cycle': f'cycler("color", {COLORMAP})',
    # Use 10pt font in plots, to match 10pt font in document
    'axes.labelsize': 8,
    'axes.linewidth': 0.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    #'figure.autolayout': True,
    'font.size': 8,
    # Lines
    'lines.linewidth': 1,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': major_tick_width,
    'xtick.minor.width': minor_tick_width,
    'ytick.major.width': major_tick_width,
    'ytick.minor.width': minor_tick_width,
    #'xtick.major.size': 20,
    #'xtick.minor.size': 20,
}

inches_per_pt = 1 / 72.27
golden_ratio = (5**.5 - 1) / 2

def default_plot_mosaic(height_fraction=1.0, width_fraction=1.0, subplots=None, sharex=False, sharey=False, style='dissertation'):
    width_pt = WIDTH_MAP[style]
    width_pt *= width_fraction

    if subplots is None:
        subplots = '''
        A
        '''

    fig_width_inch = width_pt * inches_per_pt
    fig_height_inch = height_fraction * fig_width_inch * golden_ratio 

    plt.rcParams.update(tex_fonts)

    return plt.subplot_mosaic(mosaic=subplots, figsize=(fig_width_inch, fig_height_inch), sharex=sharex, sharey=sharey)

def default_plot(height_fraction=1.0, width_fraction=1.0, subplots=None, sharex=False, sharey=False, single_column=True, style='dissertation'):
    width_pt = WIDTH_MAP[style]
    width_pt *= width_fraction

    if subplots is None:
        subplots = (1, 1)

    column_frac = 1 if single_column else 0.5

    fig_width_inch = width_pt * inches_per_pt * column_frac
    fig_height_inch = height_fraction * fig_width_inch * golden_ratio * (subplots[0] / subplots[1])

    plt.rcParams.update(tex_fonts)

    return plt.subplots(subplots[0], subplots[1], figsize=(fig_width_inch, fig_height_inch), sharex=sharex, sharey=sharey)

# Similar to tikz / pgfplot
def mix_with_white(color, percent=0.5):
    base_rgb = np.array(mc.to_rgb(color))
    white_rgb = np.array([1, 1, 1])
    mixed = (1-percent) * base_rgb + percent * white_rgb
    return tuple(mixed)

