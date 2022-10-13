import yaml, os, subprocess, shlex
from . import logging

LOGGER = logging.get_logger(__file__)


def read_yaml(filename):

    with open(filename, "r") as fobj:
        d = yaml.load(fobj, Loader=yaml.FullLoader)
    LOGGER.debug("read yaml {}".format(filename))
    return d


def is_remote(path):
    return "@" in path and ":/" in path


def robust_makedirs(path):

    if is_remote(path):
        LOGGER.info("Creating remote directory {}".format(path))
        host, path = path.split(":")
        cmd = 'ssh {} "mkdir -p {}"'.format(host, path)
        subprocess.call(shlex.split(cmd))

    elif not os.path.isdir(path):
        try:
            os.makedirs(path)
            LOGGER.info(f"Created directory {path}")
        except FileExistsError as err:
            LOGGER.error(f"already exists {path}")
