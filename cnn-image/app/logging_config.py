import logging.config
import json
import os


def setup_logging():
    config_path = os.path.join(os.path.dirname(__file__), "logging_config.json")
    with open(config_path) as f:
        config = json.load(f)
    logging.config.dictConfig(config)
