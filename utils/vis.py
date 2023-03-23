import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from utils.tools import fig2array


def draw_route(map_size: ndarray, route: ndarray,
               cmap: str = 'viridis', return_mode: str = None):
    route = route * map_size
    route = route.reshape((-1, 1, 2))

    idxs = np.array(range(route.shape[0]))
    fig = plt.figure()
    fig.patch.set_facecolor('none')
    ax = fig.add_subplot(111)
    ax.plot(route[:, 0, 0], route[:, 0, 1], '--', ms=5)

    norm = plt.Normalize(idxs[0], idxs[-1])
    segments = np.concatenate([route[:-1], route[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(idxs)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, ticks=idxs[::int(len(idxs) / 10)], label='step')

    ax.set_xlim(0, map_size[0])
    ax.set_xlabel('x')
    ax.set_ylim(0, map_size[1])
    ax.set_ylabel('y')
    ax.set_aspect(1)

    ax.grid(visible=False)

    if return_mode is not None:
        return fig
    else:
        fig.show()


def draw_routes(routes: tuple[ndarray, ndarray], return_mode: str = None):
    assert return_mode in ['plt_fig', 'fig_array', None]
    titles = ('GT', 'pred')
    cmaps = ('viridis', 'plasma')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    for i, route in enumerate(routes):
        route = route.reshape((-1, 1, 2))
        idxs = np.array(range(route.shape[0]))
        axes[i].plot(route[:, 0, 0], route[:, 0, 1], '--', ms=5)

        norm = plt.Normalize(idxs[0], idxs[-1])
        segments = np.concatenate([route[:-1], route[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmaps[i], norm=norm)
        lc.set_array(idxs)
        lc.set_linewidth(3)
        line = axes[i].add_collection(lc)
        fig.colorbar(line, ax=axes[i], ticks=idxs[::int(len(idxs) / 10)], label='step', fraction=0.05)

        axes[i].set_title(titles[i])
        axes[i].set_xlim(0, 1)
        axes[i].set_xlabel('x')
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel('y')
        axes[i].set_aspect(1)
        axes[i].grid(visible=False)

    if return_mode is None:
        fig.show()
    elif return_mode == 'plt_fig':
        return fig
    elif return_mode == 'fig_array':
        return fig2array(fig)


