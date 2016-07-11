import io
import json
import os
import unittest
import zipfile

import pandas
import requests

from estimator import Estimator


class TestEstimator(unittest.TestCase):
    def test_movie_lens_1m(self):
        with open(os.path.join(os.path.dirname(__file__), 'movie_lens_1m.json')) as config_file:
            config = json.load(config_file)
        zip_file = zipfile.ZipFile(io.BytesIO(requests.get(config['url']).content))
        data_set = pandas.read_csv(io.StringIO(zip_file.open(config['file']).read().decode('UTF-8')),
                                   sep=config['separator'], header=None, names=config['attributes'], engine='python')
        estimator = Estimator(config, data_set)
        estimator.estimate()
