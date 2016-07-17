import io
import json
import os
import zipfile

import pandas
import requests

import estimator


def main(config_path, has_header):
    with open(os.path.join(os.path.dirname(__file__), config_path)) as config_file:
        config = json.load(config_file)
    zip_file = zipfile.ZipFile(io.BytesIO(requests.get(config['url']).content))
    if has_header:
        data_set = pandas.read_csv(io.StringIO(zip_file.open(config['file']).read().decode(errors='ignore')),
                                   sep=config['separator'])
    else:
        data_set = pandas.read_csv(io.StringIO(zip_file.open(config['file']).read().decode(errors='ignore')),
                                   sep=config['separator'], header=None, names=config['attributes'], engine='python')
    print(data_set.head())
    movie_estimator = estimator.Estimator(config, data_set)
    movie_estimator.estimate()


if __name__ == '__main__':
    main('book-crossing.json', True)
