import sys

from shared.logger import Logger

sys.path.append('../workers/')


def log_instance():
    return Logger().logger_client
