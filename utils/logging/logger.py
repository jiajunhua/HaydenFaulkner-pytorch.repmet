import os
import logging
import time


def initialize_logger(save_path, run_id):
    """
    initialize a logger object
    :param save_path: the path to save the log file
    :param run_id: the id of the run
    :return: the logger
    """
    log_file = '{}_{}.log'.format(run_id, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(save_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger
