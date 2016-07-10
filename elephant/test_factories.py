import json
import os
import unittest

import factories


class TestEstimatorFactory(unittest.TestCase):
    def test_build_estimator(self):
        with open(os.path.join(os.path.dirname(__file__), 'movie_specs.json')) as config_file:
            config = json.load(config_file)
        edge_estimator_factory = factories.EstimatorFactory(config)
        edge_estimator_factory.dummy()
