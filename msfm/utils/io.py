import yaml
from . import logging

LOGGER = logging.get_logger(__file__)


def read_yaml(filename):

    with open(filename, "r") as fobj:
        d = yaml.load(fobj, Loader=yaml.FullLoader)
    LOGGER.debug("read yaml {}".format(filename))
    return d
