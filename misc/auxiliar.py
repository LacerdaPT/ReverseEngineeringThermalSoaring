import logging
import os
import sys
import collections.abc

import numpy as np
import yaml

from collections.abc import MutableMapping


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reform_dict(dictionary, t=tuple(), reform={}):
    for key, val in dictionary.items():
        t = t + (key,)
        if isinstance(val, dict):
            reform_dict(val, t, reform)
        else:
            reform.update({t: val})
        t = t[:-1]
    return reform


def config_logger(logger_instance, output_dir, verbosity, log_to_file):
    logger_instance.setLevel(logging.DEBUG)

    if isinstance(verbosity, str):
        verbosity = logging.getLevelName(verbosity.upper())

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s.%(funcName)s:%(lineno)d | %(message)s')

    if log_to_file:
        output_file = os.path.join(output_dir, 'log.log')
        file_handler = logging.FileHandler(output_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(verbosity)
    stdout_handler.setFormatter(formatter)

    logger_instance.addHandler(stdout_handler)


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def deep_dictionary_update(base_dictionary, dict_to_update_with, condition_for_update=None):
    if condition_for_update is None:
        condition_for_update = lambda _: True
    for new_key, new_value in dict_to_update_with.items():
        if isinstance(new_value, collections.abc.Mapping):
            base_dictionary[new_key] = deep_dictionary_update(base_dictionary.get(new_key, {}), new_value,
                                                              condition_for_update=condition_for_update)
        else:
            if condition_for_update(new_value):
                base_dictionary[new_key] = new_value
    return base_dictionary


def sanitize_dict_for_yaml(d):
    for key in d:
        if isinstance(d[key], collections.abc.Mapping):
            d[key] = sanitize_dict_for_yaml(d[key])
        elif isinstance(d[key], np.ndarray):
            d[key] = d[key].tolist()
        elif isinstance(d[key], (np.float64, np.float32)):
            d[key] = float(d[key])
        elif isinstance(d[key], np.int64):
            d[key] = int(d[key])

    return d
