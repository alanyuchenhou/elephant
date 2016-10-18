import pandas


def compare():
    data_sets = ['airport', 'collaboration', 'congress', 'forum', ]
    models = ['pWSBM', 'bWSBM', 'SBM', 'DCWBM', 'DCBM', 'Model R', ]
    errors = pandas.DataFrame([
        [0.0486, 0.0543, 0.0632, 0.0746, 0.0918, 0.0131, ],
        [0.0407, 0.0462, 0.0497, 0.0500, 0.0849, 0.0303, ],
        [0.0571, 0.0594, 0.0634, 0.0653, 0.1050, 0.0369, ],
        [0.0726, 0.0845, 0.0851, 0.0882, 0.0882, 0.0376, ],
    ],
        data_sets,
        models,
    )
    print(errors)
    axes = errors.plot(grid=True, xticks=range(len(data_sets)), marker='o')
    axes.set_xlabel('dataset')
    axes.set_ylabel('MSE')
    axes.legend(loc='lower right', ncol=3)
    axes.get_figure().savefig('../../../cave/link-weight-errors')


def main():
    compare()


if __name__ == '__main__':
    main()
