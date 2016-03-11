import json
import os
import unittest

import visualizer


class TestEstimatorFactory(unittest.TestCase):
    def test_plot(self):
        with open(os.path.join(os.path.dirname(__file__), 'config.json')) as config_file:
            config = json.load(config_file)
        with open(config['specs_file']) as specs_file:
            specs = json.load(specs_file)
        log_path = specs['log_path']
        figure_path = specs['figure_path']
        visualizer.plot(log_path, figure_path)
