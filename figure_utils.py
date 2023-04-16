import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def create_boxplot_figure_save_to_file(data_as_array_of_arrays, label_array, output_filename, show=False):
    fig = plt.figure(figsize=(5, 3.5))
    # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])
    ax = fig.add_subplot(111)
    # Creating plot
    bp = ax.boxplot(data_as_array_of_arrays)
    ax.set_xticklabels(label_array)
    # plt.title("Accuracy as a function of reduced dimensions")
    ax.get_xaxis().tick_bottom()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.savefig(fname=output_filename)
    if show:
        # # show plot
        plt.show()