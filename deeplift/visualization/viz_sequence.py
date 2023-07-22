"""
@Time : 2022/5/9 21:02
@Auth : zy
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))

# def plot_u(ax, base, left_edge, height, color):
#     ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.5, base + 0.5 * height], width=1, height=height,
#                                             facecolor=color, edgecolor=color))
#     ax.add_patch(
#         matplotlib.patches.Ellipse(xy=[left_edge + 0.5, base + 0.5 * height], width=0.69, height=0.7 * height,
#                                    facecolor='white', edgecolor='white'))
#     ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.5 * height], width=1.0, height=height,
#                                               facecolor='white', edgecolor='white', fill=True))
#     ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge-0.001, base + 0.5 * height],
#                                               width=0.15, height=0.5 * height, facecolor=color, edgecolor=color,
#                                               fill=True))
#     ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.849, base + 0.5 * height],
#                                               width=0.15, height=0.5 * height, facecolor=color, edgecolor=color,
#                                               fill=True))

def plot_u(ax, base, left_edge, height, color):
    # ax.add_patch(matplotlib.patches.Circle(xy=[left_edge + 0.5, base + 0.5 * height], radius=0.5*height,
    #                                         facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.5, base + 0.5 * height], width=1, height=height,
                                            facecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.5, base + 0.5 * height], width=0.7, height=0.7 * height,
                                   facecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.5 * height], width=1, height=0.5 *height,
                                              facecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.5 * height],
                                              width=0.15, height=0.5 * height, facecolor=color,
                                              fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.849, base + 0.5 * height],
                                              width=0.15, height=0.5 * height, facecolor=color,
                                              fill=True))



# default_colors = {0: 'tab:green', 1: 'tab:blue', 2: 'tab:orange', 3: 'tab:red'}
default_colors = {0: '#109648', 1: '#255C99', 2: '#F7B32B', 3: '#D62839'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_u}


def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if (array.shape[0] == 4 and array.shape[1] != 4):
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    # ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
    # ax.set_ylim(0, max_pos_height + height_padding)
    ax.set_ylim(array.min(), array.max())
    from pylab import yticks
    yticks(np.linspace(0, 0.12, 4, endpoint=True))


def plot_weights(array,
                 figsize=(20, 1),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={},
                 path=''):
    fig = plt.figure(figsize=figsize)#=========================================================================================
    ax = fig.add_subplot(111)

    # 取消边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)

    ax.spines['left'].set_visible(False)
    ax.yaxis.set_visible(False)

    plot_weights_given_ax(ax=ax, array=array,
                          height_padding_factor=height_padding_factor,
                          length_padding=length_padding,
                          subticks_frequency=subticks_frequency,
                          colors=colors,
                          plot_funcs=plot_funcs,
                          highlight=highlight)

    # plt.show()
    plt.savefig(path)