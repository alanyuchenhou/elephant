import numpy
import pandas
from sklearn import preprocessing, model_selection


def make_data_files(links_file, training_file, testing_file):
    links = pandas.read_csv(links_file, sep='\t', header=None)
    links[2] = preprocessing.MaxAbsScaler().fit_transform(numpy.log(links[2].values.reshape(-1, 1)))
    training_links, testing_links = model_selection.train_test_split(links, test_size=0.2)
    training_links.to_csv(training_file, index=False, header=False, quotechar=' ')
    testing_links.to_csv(testing_file, index=False, header=False, quotechar=' ')


def main():
    for data_set_name in ['airport', 'collaboration', 'congress', 'forum']:
        links_file = '../graph/' + data_set_name + '.tsv'
        training_file = '../data/' + data_set_name + '_training.csv'
        testing_file = '../data/' + data_set_name + '_testing.csv'
        make_data_files(links_file, training_file, testing_file)


if __name__ == '__main__':
    main()
