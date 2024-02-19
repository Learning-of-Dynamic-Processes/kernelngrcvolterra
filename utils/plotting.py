import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

def plot_data(data_list, plot_mode='1d'):
    
    """
    Plot the data variable based on the specified plot_mode.

    Parameters
    ----------
    data_list : list of array_like
        List of numpy arrays, each with shape (ndata, ndim).
    plot_mode : str 
        Options: {'1d', 'nd'}.
        If plot_mode is '1d', it will plot data for each dimension in subfigures.
        If plot_mode is 'nd', it will follow the specified conditions:
        - If ndim is 1, it will plot only one coordinate.
        - If ndim is 2, it will plot a 2D plot of one coordinate versus another one.
        - If ndim is 3, it will plot a 3D plot.
        - If ndim is higher than 3, it will plot all possible combinations for 3D plots.
    """
    
    if not isinstance(data_list, list):
        data_list = [data_list]

    colors = ['b', 'm', 'g', 'c']
    line_styles = ['-', '-.', '--', '-.']
    marker_styles = ['o', 's', '*', 'D']
    ndata, ndim = data_list[0].shape
    
    if plot_mode == '1d':
            fig, axes = plt.subplots(ndim, 1, figsize=(8, 4 * ndim))
            if ndim == 1:
                axes = [axes]  # Wrap in a list to handle 1D case
            for dim, ax in enumerate(axes):
                for i, data in enumerate(data_list):
                    color = colors[i % len(colors)]
                    line_style = line_styles[i % len(line_styles)]
                    marker_style = marker_styles[i % len(marker_styles)]
                    ax.plot(data[:, dim], label=f'Data {i + 1}, Dimension {dim + 1}', linestyle=line_style, color=color)
                ax.set_xlabel(f'Dimension {dim + 1}')
                ax.set_ylabel('Value')
                ax.set_title(f'Dimension {dim + 1} vs. Time')
                ax.legend() 
            plt.tight_layout()
            
    elif plot_mode == 'nd':
        if ndim == 1:
            plt.figure(figsize=(8, 4))
            for i, data in enumerate(data_list):
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)]
                marker_style = marker_styles[i % len(marker_styles)]
                plt.plot(data[:, 0], label=f'Data {i + 1}', linestyle=line_style, color=color)
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('1D Plot')
        elif ndim == 2:
            plt.figure(figsize=(8, 6))
            for i, data in enumerate(data_list):
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)]
                marker_style = marker_styles[i % len(marker_styles)]
                plt.plot(data[:, 0], data[:, 1], label=f'Data {i + 1}', linestyle=line_style, color=color)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('2D Plot')
        elif ndim == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            for i, data in enumerate(data_list):
                color = colors[i % len(colors)]
                line_style = line_styles[i % len(line_styles)]
                marker_style = marker_styles[i % len(marker_styles)]
                ax.plot(data[:, 0], data[:, 1], data[:, 2], label=f'Data {i + 1}', linestyle=line_style, color=color)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            ax.set_title('3D Plot')
            ax.legend()
        else:
            comb_3d = list(combinations(range(ndim), 3))
            ncomb = len(comb_3d)
            nrows = int(np.ceil(ncomb / 2))
            fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows))
            for i, comb_i in enumerate(comb_3d):
                row = i // 2
                col = i % 2
                ax = axes[row, col]
                for i, data in enumerate(data_list):
                    color = colors[i % len(colors)]
                    line_style = line_styles[i % len(line_styles)]
                    marker_style = marker_styles[i % len(marker_styles)]
                    ax.plot(data[:, comb_i[0]], data[:, comb_i[1]], data[:, comb_i[2]], linestyle=line_style, color=color)
                ax.set_xlabel(f'Dimension {comb_i[0] + 1}')
                ax.set_ylabel(f'Dimension {comb_i[1] + 1}')
                ax.set_zlabel(f'Dimension {comb_i[2] + 1}')
                ax.set_title(f'3D Plot: Dimensions {comb_i[0] + 1}, {comb_i[1] + 1}, {comb_i[2] + 1}')
                ax.legend()
            plt.tight_layout()
    plt.legend()
    plt.show()

def plot_data_distributions(data_list):
    
    """
    Superpose KDE plots for each dimension across all elements in data_list.

    Parameters:
    - data_list: List of numpy arrays, each with shape (ndata, ndim).

    This function creates subplots for each dimension and superposes KDE plots for all data elements in data_list.
    """
    
    if not isinstance(data_list, list):
        data_list = [data_list]

    ndata, ndim = data_list[0].shape
    colors = ['b', 'r', 'g', 'c']
    
    fig, axs = plt.subplots(ndim, figsize=(8, 4 * ndim))
    fig.tight_layout(pad=2)

    for dim in range(ndim):
        for i, data in enumerate(data_list):
            color = colors[i % len(colors)]
            sns.kdeplot(data[:, dim], color=color, fill=True, label=f'Data {i + 1}', ax=axs[dim])
            
        axs[dim].set(xlabel=f"Dimension {dim + 1}")
        axs[dim].legend()

    plt.show()
    
