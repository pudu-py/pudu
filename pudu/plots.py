import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
import numpy as np

def plot(feature, image, axis=None, show_data=True, title='Importance', 
        xlabel='Feature', ylabel='Intensity', xticks=None, yticks=[], cmap='Greens',
        font_size=15, figsize=(14, 4), padding=(0, 0, 1, 1), cbar_pad=0.05, vmin=None,
        vmax=None, xlims=None):
    """
    Easy plot function for `importance`, `speed`, or `synergy`. It shows the analyzed
        feature `feature` with a colormap overlay indicating the result along with
        a colorbar. Works for both vectors and images.

    :type feature: list
    :param feature: feature analyzed or any that the user whant to plot against.
        Normally you want it to be `self.x`.

    :type image: list
    :param image: Result you want to plot. Either `self.imp`, `self.syn`, etc...

    :type axis: list
    :param axis: X-axis for the plot. If `None`, it will show the pixel count.
    
    :type show_data: bool
    :param show_data: . Default is `True`.

    :type title: str
    :param title: Title for the plot. Default is `Importnace`.
        
    :type xlabel: str
    :param xlabel: X-axis title. Default is `Feature`.
        
    :type ylabel: str
    :param ylabel: Y-axis title. Default is `Intensity`
    
    :type xticks: list
    :param xticks: Ticks to display on the graph. Default is `None`.

    :type yticks: list
    :param yticks: Ticks to display on the graph. Default is `[]`.

    :type cmap: string
    :param cmap: colormap for `image` according to the availability in `matplotlib`.
        Default is `plasma`.

    :type font_size: int
    :param font_size: Font size for all text in the plot. Default is `15`.
        
    :type figsize: tuple
    :param figsize: Size of the figure. Default is `(14, 4)`.
    """

    dims = np.array(feature).shape

    image = np.array(image)[0,:,:,0]
    feature = np.array(feature)[0,:,:,0]

    if vmin is None:
        vmin = np.min(image)
        
    if vmax is None:
        vmax = np.max(image)

    if dims[1] > 1:
        rows, cols = dims[1], dims[2]
        ext = [0, cols, 0, rows]
    else:
        rows, cols = 1, len(image)
                
        if axis is None:
            axis = [i for i in range(len(feature[0]))]
            ext = [0, len(feature[0]), min(feature[0]), max(feature[0])]
        else:
            ext = [min(axis), max(axis), min(feature[0]), max(feature[0])]
    
    plt.rc('font', size=font_size)
    plt.subplots_adjust(left=padding[0], bottom=padding[1], right=padding[2], top=padding[3])
    plt.figure(figsize=figsize)
    if dims[1] > 1 and show_data:
        plt.imshow(feature, cmap='binary', aspect="auto", 
                    interpolation='nearest', extent=ext, alpha=1,
                    vmin=vmin, vmax=vmax)
    elif dims[1] == 1 and show_data:
        plt.plot(axis, feature[0], 'k')
    plt.imshow(image, cmap=cmap, aspect="auto", 
                interpolation='nearest', extent=ext, alpha=0.5,
                vmin=vmin, vmax=vmax)
    
    plt.title(title) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(yticks)
    
    if xlims:
        plt.xlim(xlims[0], xlims[1])

    if xticks:
        plt.xticks(axis, xticks, rotation='vertical')
    
    plt.tight_layout()
    plt.colorbar(pad=cbar_pad, ticks=ticker.LinearLocator(numticks=5))
    plt.show()  


def feature_report(aso, plot=True, print_report=False, sort=True, title='Feature Report', 
                    ylabel='Counts', xlabel='Feature', fig_size=(7,5), rot_ticks=45, 
                    bar_width=0.9, background_color='thistle', bar_color='purple', 
                    x_lims=(-0.5, -0.61), borders=True, font_size=14, 
                    show_top=None):
    """
    Generates a report of units activation in the form of a plot or printed text.

    :type aso: list
    :param aso: A list of unit index and coordinates for unit activations.

    :type plot: bool
    :param plot: If True, a plot of the unit activations is displayed. 

    :type print_report: bool
    :param print_report: If True, the report is printed in text form in the console.

    :type sort: bool
    :param sort: If True, the results are sorted by the count of activations.

    :type title: str
    :param title: The title of the plot. Default is 'Unit Report'.

    :type ylabel: str
    :param ylabel: The label for the y-axis of the plot. Default is 'Counts'.

    :type xlabel: str
    :param xlabel: The label for the x-axis of the plot. Default is 'Feature'.

    :type fig_size: tuple
    :param fig_size: The size of the figure for the plot. Default is (10, 5).

    :type rot_ticks: int
    :param rot_ticks: The rotation angle of the x-axis tick labels. Default is 90.
    
    :type bar_width: float
    :param bar_width: The width of the bars in the plot. Default is 0.9.
    
    :type background_color: str
    :param background_color: The background color of the plot. Default is 'thistle'.
    
    :type bar_color: str
    :param bar_color: The color of the bars in the plot. Default is 'purple'.
    
    :type x_lims: tuple
    :param x_lims: The limits for the x-axis. Default is (-0.5, -0.61).
    
    :type borders: bool
    :param borders: If False, the borders of the plot are not displayed.
    
    :type font_size: int
    :param font_size: The size of the font used in the plot. Default is 14.
    
    :type show_top: int
    :param show_top: If given, only the top 'show_top' coordinates are displayed. Default is None.
    """
    labels = [i for i in range(len(aso))]

    if sort:
        aso, labels = zip(*sorted(zip(aso, labels), reverse=True))

    if print_report:
        for i, j in zip(labels, aso):
            print(f"The feature {i} activates {j} units.")
    
    if show_top == 0 or show_top == 'all' or (show_top is None):
            show_top = len(aso)

    if plot:
        labels, aso = labels[:show_top], aso[:show_top]
        labels = [str(i) for i in labels]
        bar_plot(labels, aso, None, font_size, fig_size, bar_width, bar_color, 
                borders, title, xlabel, rot_ticks, x_lims, ylabel, background_color)


def unit_report(aso, plot=True, print_report=False, sort=True, title='Unit Report', 
                    ylabel='Counts', xlabel='Unit', fig_size=(7,5), rot_ticks=90, 
                    bar_width=0.9, background_color='bisque', bar_color='darkorange', 
                    x_lims=(-0.5, -0.61), borders=True, font_size=14, show_top=None):
    """
    Generates a report of unit activation in the form of a plot or printed text.

    :type aso: list
    :param aso: A list of unit index and coordinates for unit activations.

    :type plot: bool
    :param plot: If True, a plot of the unit activations is displayed. 

    :type print_report: bool
    :param print_report: If True, the report is printed in text form in the console.

    :type sort: bool
    :param sort: If True, the results are sorted by the count of activations.

    :type title: str
    :param title: The title of the plot. Default is 'Unit Report'.

    :type ylabel: str
    :param ylabel: The label for the y-axis of the plot. Default is 'Counts'.

    :type xlabel: str
    :param xlabel: The label for the x-axis of the plot. Default is 'Feature'.

    :type fig_size: tuple
    :param fig_size: The size of the figure for the plot. Default is (10, 5).

    :type rot_ticks: int
    :param rot_ticks: The rotation angle of the x-axis tick labels. Default is 90.
    
    :type bar_width: float
    :param bar_width: The width of the bars in the plot. Default is 0.9.
    
    :type background_color: str
    :param background_color: The background color of the plot. Default is 'thistle'.
    
    :type bar_color: str
    :param bar_color: The color of the bars in the plot. Default is 'purple'.
    
    :type x_lims: tuple
    :param x_lims: The limits for the x-axis. Default is (-0.5, -0.61).
    
    :type borders: bool
    :param borders: If False, the borders of the plot are not displayed.
    
    :type font_size: int
    :param font_size: The size of the font used in the plot. Default is 14.
    
    :type show_top: int
    :param show_top: If given, only the top 'show_top' coordinates are displayed. Default is None.
    """

    labels = [i for i in range(len(aso))]

    if sort:
        aso, labels = zip(*sorted(zip(aso, labels), reverse=True))

    if print_report:
        for i, j in zip(labels, aso):
            print(f"The unit {i} activates {j} times.")
    
    if show_top == 0 or show_top == 'all' or (show_top is None):
            show_top = len(aso)

    if plot:
        labels, aso = labels[:show_top], aso[:show_top]
        labels = [str(int(i)) for i in labels]
        
        bar_plot(labels, aso, None, font_size, fig_size, bar_width, bar_color, 
                borders, title, xlabel, rot_ticks, x_lims, ylabel, background_color)


def relate_report(x, y, z, plot=True, print_report=False, sort=True, title='Units-Feature', 
                    ylabel='Counts', xlabel='Unit Index', fig_size=(7,5), rot_ticks=0, 
                    bar_width=0.9, background_color='palegreen', bar_color='green', 
                    x_lims=(-0.5, -0.61), borders=True, font_size=14, show_top=None):
    """
    Generates a report of unit activation in the form of a plot or printed text.

    :type x: list or numpy array
    :param x: List or array of unit indices.
    
    :type y: list or numpy array
    :param y: List or array of counts of unit activations.
    
    :type z: list or numpy array
    :param z: List or array of features for unit activations.

    :type plot: bool
    :param plot: If True, a plot of the unit activations is displayed. 

    :type print_report: bool
    :param print_report: If True, the report is printed in text form in the console.

    :type sort: bool
    :param sort: If True, the results are sorted by the count of activations.

    :type title: str
    :param title: The title of the plot. Default is 'Unit Report'.

    :type ylabel: str
    :param ylabel: The label for the y-axis of the plot. Default is 'Counts'.

    :type xlabel: str
    :param xlabel: The label for the x-axis of the plot. Default is 'Feature'.

    :type fig_size: tuple
    :param fig_size: The size of the figure for the plot. Default is (10, 5).

    :type rot_ticks: int
    :param rot_ticks: The rotation angle of the x-axis tick labels. Default is 90.
    
    :type bar_width: float
    :param bar_width: The width of the bars in the plot. Default is 0.9.
    
    :type background_color: str
    :param background_color: The background color of the plot. Default is 'thistle'.
    
    :type bar_color: str
    :param bar_color: The color of the bars in the plot. Default is 'purple'.
    
    :type x_lims: tuple
    :param x_lims: The limits for the x-axis. Default is (-0.5, -0.61).
    
    :type borders: bool
    :param borders: If False, the borders of the plot are not displayed.
    
    :type font_size: int
    :param font_size: The size of the font used in the plot. Default is 14.
    
    :type show_top: int
    :param show_top: If given, only the top 'show_top' coordinates are displayed. Default is None.
    """

    if sort:
        lists = list(zip(x, y, z))
        lists.sort(key=lambda i: i[1], reverse=True)
        x, y, z = zip(*lists)

    if show_top == 0 or show_top == 'all' or (show_top is None):
            show_top = len(x)

    if print_report:
        for i, j, k in zip(x[:show_top], y[:show_top], z[:show_top]):
            print(f"The unit {i} activates the most with feature {k}, {j} times.")
    
    if plot:    
        x, y, z = x[:show_top], y[:show_top], z[:show_top]
        x = [str(int(i)) for i in x]
       

        value_count = {}
        for i in range(len(x)):
            value = x[i]
            if value not in value_count:
                value_count[value] = 0
            else:
                value_count[value] += 1

            x[i] = f'{value}-{value_count[value]}'


        bar_plot(x, y, z, font_size, fig_size, bar_width, bar_color, 
                borders, title, xlabel, rot_ticks, x_lims, ylabel, background_color)


def bar_plot(x, y, z, font_size, fig_size, bar_width, bar_color, borders, 
             title, xlabel, rot_ticks, x_lims, ylabel, background_color):
    """
    To avoid redundancy.
    """
    # min_count, max_count = 1, max(counts)
    # if max_count%2 == 0:
    #     min_count = 0
    # y_ticks = np.linspace(min_count, max_count, num=int((max_count-min_count)*0.5+1))

    mpl.rcParams['font.size'] = font_size
    plt.figure(figsize=fig_size)
    bars = plt.bar(x, y, width=bar_width, color=bar_color)

    if z is not None:
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2 - font_size/50, bar.get_width()/1, z[i], va='bottom', rotation=90) # va='bottom' to align text # va='bottom' to align text
            print(bar.get_x(), bar.get_width())
    if not borders:
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                            labelbottom=True, left=False, right=False, labelleft=True)
        
    plt.title(title)

    plt.xlabel(xlabel)
    plt.xticks(rotation=rot_ticks)
    plt.xlim(bar_width*x_lims[0], len(x)+bar_width*x_lims[1])
    
    plt.ylabel(ylabel)
    # plt.yticks(y_ticks)
    plt.ylim(0, max(y))

    plt.gca().set_facecolor(background_color)
    plt.tight_layout()
    plt.show()
