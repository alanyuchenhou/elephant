import pandas


def compare(data_sets):
    models = ['pWSBM', 'bWSBM', 'SBM', 'DCWBM', 'node2vec', 'LLE', 'Model R', ]
    errors = pandas.DataFrame([
        [0.0486, 0.0543, 0.0632, 0.0746, 0.0171, 0.0170, 0.0114, ],
        [0.0407, 0.0462, 0.0497, 0.0500, 0.0614, 0.0576, 0.0327, ],
        [0.0571, 0.0594, 0.0634, 0.0653, 0.0386, 0.0448, 0.0365, ],
        [0.0726, 0.0845, 0.0851, 0.0882, 0.0312, 0.0304, 0.0298, ],
    ],
        data_sets,
        models,
    )
    print(errors)
    axes = errors.plot.bar(rot=0, figsize=(16, 8,), grid=True, )
    axes.set_xlabel('dataset')
    axes.set_ylabel('mean squared error')
    axes.legend(loc='upper left', ncol=6, )
    axes.get_figure().savefig('../log/link-weight-errors')


def plot_running_time(data_sets):
    attribute = 'running_time'
    times = pandas.DataFrame([0.880, 2.469, 0.974, 2.521, 0.889, 1.151], data_sets, [attribute])
    print(times)
    axes = times.plot.bar(rot=0)
    axes.set_xlabel('data_set')
    axes.set_ylabel(attribute + '(sec)')
    axes.get_figure().savefig('../log/' + attribute)


def plot_errors(data_set, parameter):
    node2vec_errors = pandas.read_csv('../../node2vec/log/' + data_set + '/errors.csv')
    model_r_errors = pandas.read_csv('../../model_r/log/' + data_set + '/errors.csv')
    errors = pandas.DataFrame({
        'node2vec': node2vec_errors['error'].tolist(),
        'ModelR': model_r_errors['error'].tolist(),
    },
        index=model_r_errors[parameter].tolist(),
    )
    axes = errors.plot(ylim=(0, 0.07))
    axes.set_xlabel(parameter)
    axes.set_ylabel('mean_squared_error')
    axes.set_title(data_set)
    axes.get_figure().savefig('../log/' + parameter + '_' + data_set)


def main():
    data_sets = ['airport', 'authors', 'collaboration', 'facebook', 'congress', 'forum']
    # compare(data_sets)
    plot_running_time(data_sets)
    # for data_set in data_sets:
    #     plot_errors(data_set, 'num_hidden_layers')


if __name__ == '__main__':
    main()
