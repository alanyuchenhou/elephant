import pandas


def compare():
    data_sets = ['airport', 'collaboration', 'congress', 'forum', ]
    # models = ['pWSBM', 'bWSBM', 'SBM', 'DCWBM', 'node2vec', 'LLE', 'Model R', ]
    # errors = pandas.DataFrame([
    #     [0.0486, 0.0543, 0.0632, 0.0746, 0.0171, 0.0170, 0.0114, ],
    #     [0.0407, 0.0462, 0.0497, 0.0500, 0.0614, 0.0576, 0.0327, ],
    #     [0.0571, 0.0594, 0.0634, 0.0653, 0.0386, 0.0448, 0.0365, ],
    #     [0.0726, 0.0845, 0.0851, 0.0882, 0.0312, 0.0304, 0.0298, ],
    # ],
    #     data_sets,
    #     models,
    # )
    models = ['pwsbm', 'sbm', 'model-s(node2vec)', 'model-s(lle)', 'model-s(model-r)', ]
    errors = pandas.DataFrame([
        [0.0486, 0.0632, 0.0171, 0.0170, 0.0114, ],
        [0.0407, 0.0497, 0.0614, 0.0576, 0.0327, ],
        [0.0571, 0.0634, 0.0398, 0.0402, 0.0365, ],
        [0.0726, 0.0851, 0.0312, 0.0294, 0.0298, ],
    ],
        data_sets,
        models,
    )
    print(errors)
    axes = errors.plot.bar(rot=0, figsize=(16, 8,), grid=True, )
    axes.set_xlabel('dataset')
    axes.set_ylabel('mean squared error')
    axes.legend(loc='upper left', ncol=6, )
    axes.get_figure().savefig('../log/weight-errors')


def plot_rating_errors():
    data_sets = ['MovieLens100K', 'MovieLens1M', 'Epinions', 'MovieTweetings']
    models = ['PCC', 'WPCC', 'SPCC', 'MPCC', 'Model R', ]
    errors = pandas.DataFrame([
        [0.83, 0.82, 0.83, 0.82, 0.69, ],
        [0.83, 0.81, 0.83, 0.79, 0.65, ],
        [1.00, 1.02, 1.00, 0.93, 0.76, ],
        [1.38, 1.32, 1.33, 1.26, 1.15, ],
    ],
        data_sets,
        models,
    )
    print(errors)
    axes = errors.plot.bar(rot=0, figsize=(16, 8,), grid=True, )
    axes.set_xlabel('dataset')
    axes.set_ylabel('mean absolute error')
    axes.legend(loc='upper left', ncol=6, )
    axes.get_figure().savefig('../log/rating-errors')


def plot_running_time(data_sets):
    attribute = 'running_time'
    times = pandas.DataFrame([0.880, 2.469, 0.974, 2.521, 0.889, 1.151], data_sets, [attribute])
    print(times)
    axes = times.plot.bar(rot=0)
    axes.set_xlabel('data_set')
    axes.set_ylabel(attribute + '(sec)')
    axes.get_figure().savefig('../log/' + attribute)


def plot_errors(data_set, parameter):
    models = ['lle', 'node2vec', 'model_r']
    errors = pandas.DataFrame({
        model: pandas.read_csv(
            '../../' + model + '/log/' + data_set + '/errors.csv'
        )['error'].tolist() for model in models
    },
        index=pandas.read_csv('../../model_r/log/' + data_set + '/errors.csv')[parameter].tolist(),
    )
    axes = errors.plot(ylim=(0, 0.07))
    axes.set_xlabel(parameter)
    axes.set_ylabel('mean_squared_error')
    axes.set_title(data_set)
    axes.get_figure().savefig('../log/' + parameter + '_' + data_set)


def main():
    compare()
    # plot_rating_errors()
    data_sets = ['airport', 'authors', 'collaboration', 'facebook', 'congress', 'forum']
    # plot_running_time(data_sets)
    # for data_set in data_sets:
    #     plot_errors(data_set, 'units_per_layer')


if __name__ == '__main__':
    main()
