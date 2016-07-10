import json
import os
import unittest

import visualizer


class TestEstimatorFactory(unittest.TestCase):
    def test_plot(self):
        with open(os.path.join(os.path.dirname(__file__), 'movie_specs.json')) as config_file:
            config = json.load(config_file)
        log_path = config['log_path']
        figure_path = config['figure_path']
        visualizer.plot(log_path, figure_path)
