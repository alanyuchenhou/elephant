import pandas


def plot(log_path, figure_path):
    trace = pandas.read_table(log_path)
    axis = trace.plot()
    axis.set_xlabel('epoch')
    axis.set_ylabel('MAE(mean absolute error)')
    axis.get_figure().savefig(figure_path)
