import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

from mpl_toolkits.mplot3d import Axes3D   # for 3D plotting


class UniversalPlotter:
    """
    Universal Plotter Class

    Supports:
    - Pandas DataFrame
    - NumPy arrays
    - Python list / tuple

    Plot Types:
    - line
    - scatter
    - bar
    - histogram
    - heatmap
    - 3D scatter
    - 3D line
    - live plot

    Extra Features:
    - optional saving feature (constructor argument)
    - subplot support
    """

    def __init__(
        self,
        figsize=(10, 6),
        save_plot=False,
        save_dir="saved_plots",
        save_format="png",
        dpi=300
    ):
        self.figsize = figsize
        self.save_plot = save_plot
        self.save_dir = save_dir
        self.save_format = save_format
        self.dpi = dpi

        if self.save_plot:
            os.makedirs(self.save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # SAVE FUNCTION
    # ---------------------------------------------------------
    def _save_figure(self, fig, filename=None):
        """Save figure automatically if save_plot enabled."""

        if not self.save_plot:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plot_{timestamp}.{self.save_format}"
        else:
            if not filename.endswith(f".{self.save_format}"):
                filename = f"{filename}.{self.save_format}"

        save_path = os.path.join(self.save_dir, filename)
        fig.savefig(save_path, dpi=self.dpi)
        print(f"✅ Plot saved at: {save_path}")

    # ---------------------------------------------------------
    # DATA PROCESSING
    # ---------------------------------------------------------
    def _process_data(self, data, x=None, y=None):
        """
        Converts input data into x_data and y_data.
        """

        if isinstance(data, pd.DataFrame):
            cols = list(data.columns)

            if isinstance(x, str) and x in cols:
                x_data = data[x].values
            else:
                x_data = None

            if isinstance(y, str) and y in cols:
                y_data = data[y].values
            else:
                y_data = None

            if x_data is None or y_data is None:
                if len(cols) == 1:
                    x_data = np.arange(len(data))
                    y_data = data[cols[0]].values
                elif len(cols) >= 2:
                    x_data = data[cols[0]].values
                    y_data = data[cols[1]].values
                else:
                    raise ValueError("DataFrame has no usable columns.")

            return x_data, y_data

        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                x_data = np.arange(len(data))
                y_data = data
                return x_data, y_data

            elif data.ndim == 2:
                if data.shape[1] == 2:
                    return data[:, 0], data[:, 1]
                elif data.shape[0] == 2:
                    return data[0, :], data[1, :]
                else:
                    raise ValueError(
                        f"2D array must be (N,2) or (2,N). Got {data.shape}"
                    )

            else:
                raise ValueError("Only 1D or 2D arrays supported.")

        elif isinstance(data, (list, tuple)):
            y_data = np.array(data)
            x_data = np.arange(len(y_data))
            return x_data, y_data

        else:
            raise ValueError("Unsupported data type.")

    # ---------------------------------------------------------
    # 3D DATA PROCESSING
    # ---------------------------------------------------------
    def _process_3d_data(self, data, x=None, y=None, z=None):
        """
        Convert input data into x, y, z arrays.
        """

        if isinstance(data, pd.DataFrame):
            cols = list(data.columns)

            if x in cols and y in cols and z in cols:
                return data[x].values, data[y].values, data[z].values

            if len(cols) >= 3:
                return data[cols[0]].values, data[cols[1]].values, data[cols[2]].values

            raise ValueError("DataFrame must have at least 3 columns for 3D plotting.")

        elif isinstance(data, np.ndarray):
            if data.ndim == 2 and data.shape[1] >= 3:
                return data[:, 0], data[:, 1], data[:, 2]

            raise ValueError("NumPy array must be shape (N,3) or more for 3D plotting.")

        else:
            raise ValueError("3D plot supports only DataFrame or NumPy array input.")

    # ---------------------------------------------------------
    # AXES SETUP
    # ---------------------------------------------------------
    def _setup_axes(self, ax=None, title=None, xlabel=None, ylabel=None, is_3d=False):
        """
        Setup axes. If ax is None -> create new figure and axes.
        """

        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            if is_3d:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        if title:
            ax.set_title(title)

        if xlabel:
            ax.set_xlabel(xlabel)

        if ylabel:
            ax.set_ylabel(ylabel)

        if not is_3d:
            ax.grid(True)

        return fig, ax

    # ---------------------------------------------------------
    # MAIN PLOT FUNCTION
    # ---------------------------------------------------------
    def plot(
        self,
        data,
        kind="line",
        x=None,
        y=None,
        label=None,
        color=None,
        marker=None,
        markersize=6,
        alpha=1.0,
        linewidth=2,
        title=None,
        xlabel=None,
        ylabel=None,
        ax=None,
        save_name=None,
        bins=10,
        **kwargs
    ):
        """
        General Plot Function.

        kind:
        - line
        - scatter
        - bar
        - hist
        - heatmap
        """

        # ---------------- Heatmap ----------------
        if kind == "heatmap":
            if isinstance(data, pd.DataFrame):
                data_array = data.values
            elif isinstance(data, np.ndarray):
                data_array = data
            else:
                raise ValueError("Heatmap supports only DataFrame or NumPy array.")

            fig, ax = self._setup_axes(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)
            cax = ax.imshow(data_array, aspect="auto")
            fig.colorbar(cax)

            self._save_figure(fig, save_name)
            return fig, ax

        # ---------------- Normal 2D plots ----------------
        x_data, y_data = self._process_data(data, x, y)
        fig, ax = self._setup_axes(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel)

        if kind == "line":
            ax.plot(
                x_data, y_data,
                label=label,
                color=color,
                marker=marker,
                markersize=markersize,
                alpha=alpha,
                linewidth=linewidth,
                **kwargs
            )

        elif kind == "scatter":
            ax.scatter(
                x_data, y_data,
                label=label,
                color=color,
                marker=marker,
                s=markersize * 10,
                alpha=alpha,
                **kwargs
            )

        elif kind == "bar":
            ax.bar(x_data, y_data, alpha=alpha, **kwargs)

        elif kind == "hist":
            ax.hist(y_data, bins=bins, alpha=alpha, **kwargs)

        else:
            raise ValueError("Invalid kind. Use line/scatter/bar/hist/heatmap")

        if label:
            ax.legend()

        self._save_figure(fig, save_name)
        return fig, ax

    # ---------------------------------------------------------
    # 3D PLOT FUNCTION
    # ---------------------------------------------------------
    def plot_3d(
        self,
        data,
        kind="scatter",
        x=None,
        y=None,
        z=None,
        title="3D Plot",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        ax=None,
        save_name=None,
        **kwargs
    ):
        """
        3D Plot Function

        kind:
        - scatter
        - line
        """

        x_data, y_data, z_data = self._process_3d_data(data, x, y, z)

        fig, ax = self._setup_axes(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, is_3d=True)
        ax.set_zlabel(zlabel)

        if kind == "scatter":
            ax.scatter(x_data, y_data, z_data, **kwargs)

        elif kind == "line":
            ax.plot(x_data, y_data, z_data, **kwargs)

        else:
            raise ValueError("3D kind must be 'scatter' or 'line'.")

        self._save_figure(fig, save_name)
        return fig, ax

    # ---------------------------------------------------------
    # SUBPLOTS CREATION
    # ---------------------------------------------------------
    def create_subplots(self, rows=1, cols=1, figsize=None):
        if figsize is None:
            figsize = self.figsize

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        return fig, axes

    # ---------------------------------------------------------
    # LIVE PLOT
    # ---------------------------------------------------------
    def live_plot(self, data_generator, interval=200, title="Live Plot", xlabel="X", ylabel="Y"):
        """
        Live plotting using matplotlib.animation.
        Generator yields:
        - y_value
        OR
        - (x_value, y_value)
        """

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        x_data, y_data = [], []
        line, = ax.plot([], [], lw=2)

        def update(frame):
            if isinstance(frame, (tuple, list)) and len(frame) == 2:
                x_val, y_val = frame
            else:
                x_val = len(x_data)
                y_val = frame

            x_data.append(x_val)
            y_data.append(y_val)

            line.set_data(x_data, y_data)
            ax.relim()
            ax.autoscale_view()
            return line,

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=data_generator,
            interval=interval,
            blit=False,
            cache_frame_data=False
        )

        plt.show()
        return ani


# ---------------------------------------------------------
# TESTING CODE
# ---------------------------------------------------------
if __name__ == "__main__":

    # Enable saving in constructor
    plotter = UniversalPlotter(save_plot=True, save_dir="plots_output")

    # DataFrame Test
    df = pd.DataFrame({
        "X": np.arange(50),
        "Y": np.random.randn(50)
    })

    print("Testing Line Plot...")
    plotter.plot(df, kind="line", x="X", y="Y", title="Line Plot", label="line", save_name="line_plot")
    plt.show()

    print("Testing Scatter Plot...")
    plotter.plot(df, kind="scatter", x="X", y="Y", title="Scatter Plot", save_name="scatter_plot")
    plt.show()

    print("Testing Histogram Plot...")
    plotter.plot(df["Y"], kind="hist", title="Histogram Plot", bins=15, save_name="hist_plot")
    plt.show()

    # Heatmap Test
    heatmap_data = np.random.rand(20, 20)
    print("Testing Heatmap...")
    plotter.plot(heatmap_data, kind="heatmap", title="Heatmap Plot", save_name="heatmap_plot")
    plt.show()

    # 3D Test
    print("Testing 3D Scatter Plot...")
    data_3d = np.random.rand(100, 3)
    plotter.plot_3d(data_3d, kind="scatter", title="3D Scatter", save_name="3d_scatter")
    plt.show()
