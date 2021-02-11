import logging


def string_to_verbosity(level):
    levelconversion = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levelconversion[level]


def set_verbosity(loglevel):
    logger = logging.getLogger('root')
    logger.setLevel(string_to_verbosity(loglevel))


def get_logger():
    logger = logging.getLogger('root')
    FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
    logging.basicConfig(format=FORMAT)
    return logger
