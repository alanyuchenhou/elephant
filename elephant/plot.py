import pandas


def compare():
    data_sets = ['Airport', 'Collaboration', 'Congress', 'Forum', ]
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
    axes.get_figure().savefig('../log/error_vs_' + parameter + '_' + data_set)


def main():
    # compare()
    for data_set in ['airport', 'collaboration', 'congress', 'forum', ]:
        plot_errors(data_set, 'dimension')


if __name__ == '__main__':
    main()
