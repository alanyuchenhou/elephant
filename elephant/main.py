import io
import json
import os
from bz2 import BZ2File
from zipfile import ZipFile

import pandas
import requests

from elephant.estimator import Estimator


def main(data_set_name):
    with open(os.path.join(os.path.dirname(__file__), data_set_name + '.json')) as specs_file:
        specs = json.load(specs_file)
    data_file = None
    compression = specs['compression']
    if compression == 'bz2':
        binary_file = BZ2File(io.BytesIO(requests.get(specs['url']).content))
        data_file = io.StringIO(binary_file.read().decode(errors='ignore'))
    elif compression == 'zip':
        binary_file = ZipFile(io.BytesIO(requests.get(specs['url']).content))
        data_file = io.StringIO(binary_file.open(specs['file']).read().decode(errors='ignore'))
    elif compression == 'none':
        data_file = specs['url']
    data_set = pandas.read_csv(data_file, sep=specs['separator'], engine=specs['engine'], skiprows=1)
    if data_set_name == 'book-crossing':
        data_set = data_set.ix[data_set['Book-Rating'] != 0]
    print(data_set.head())
    with open(os.path.join(os.path.dirname(__file__), 'neural-net.json')) as config_file:
        config = json.load(config_file)
    x = data_set.ix[:, :2].values
    estimator = Estimator(config, x)
    y = data_set.ix[:, 2].values
    print('testing_error =', estimator.estimate(specs['test_size'], config['batch_size'], config['learning_rate'], y))


if __name__ == '__main__':
    # main('book-crossing')
    # main('movie-lens-100k')
    # main('movie-lens-1m')
    # main('e-pinions')
    main('movie-tweeting')
