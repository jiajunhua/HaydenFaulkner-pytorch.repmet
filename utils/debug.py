import os
from config.config import config


def set_working_dir():
    """
    change to the working dir of pytorch.repmet

    :return: string of the new working directory
    """
    cwd = os.getcwd()
    nwd = cwd[:cwd.find(config.project)+len(config.project)]
    os.chdir(nwd)
    return nwd