#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import sys
import yaml
import logging.config


def init_logging(config_path='config/logging_config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read())
        logging.config.dictConfig(config)
    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        logging.basicConfig(level=logging.DEBUG)
