import pandas
from scipy import stats


def t_test():
    data_sets = ['Airport', 'Collaboration', 'Congress', 'Forum', ]
    models = ['pWSBM_m', 'pWSBM_s', 'ModelR_m', 'ModelR_s', 'sample_size', ]
    errors = pandas.DataFrame([
        [0.0486, 0.0006, 0.0131, 0.001, 25, ],
        [0.0407, 0.0001, 0.0303, 0.001, 25, ],
        [0.0571, 0.0004, 0.0369, 0.003, 25, ],
        [0.0726, 0.0003, 0.0376, 0.001, 25, ],
    ],
        data_sets,
        models,
    )
    print(errors)
    errors['reduction'] = errors.apply(lambda record: (record[0] - record[2]) / record[0], axis=1, )
    errors['p_value'] = errors.apply(lambda record: stats.ttest_ind_from_stats(
        record[0], record[1], record[4], record[2], record[3], record[4], )[1], axis=1, )
    return errors


def main():
    print(t_test())


if __name__ == '__main__':
    main()
