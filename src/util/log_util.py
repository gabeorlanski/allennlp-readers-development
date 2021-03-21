import os
import logging
from logging.handlers import RotatingFileHandler
import tqdm

from typing import List, Dict, Iterable


class TQDMLoggingHandler(logging.Handler):
    """
    Console Handler that works with tqdm.
    All credit goes to: https://stackoverflow.com/questions/38543506/change-logging-print
    -function-to-tqdm-write-so-logging-doesnt-interfere-wit
    """

    def __init__(self, level=logging.NOTSET, fmt: logging.Formatter = None):
        super().__init__(level)
        self.setFormatter(
            fmt if fmt is not None else logging.Formatter('%(levelname)8s: %(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class CompactFileHandler(RotatingFileHandler):
    """
    Wrapper to reduce repetition
    """

    def __init__(self, file: str, level: int = logging.NOTSET, fmt: logging.Formatter = None):
        """
        Initialize the handler
        Args:
            file: file to write to
            level: The logging level
            fmt: Logging formatter
        """
        # Clear file
        with open(file, 'w', encoding='utf-8') as f:
            f.write('')
        super(CompactFileHandler, self).__init__(file, mode='w', encoding='utf-8',
                                                 maxBytes=10485760, backupCount=3)
        self.setLevel(level)
        self.setFormatter(
            fmt if fmt is not None else logging.Formatter('%(levelname)8s: %(message)s'))


def setupLoggers(name: str, log_path: str = None, verbose: bool = False,
                 debug: bool = False) -> Iterable[logging.Logger]:
    """
    Setup the logger
    Args:
        name: Name of the logger
        log_path: Path to directory where the logs will be saved
        verbose: Enable Verbose
        debug: Enable Debug
    Returns:
        The loggers
    """
    # Load in the default paths for log_path
    log_path = os.path.join(log_path, 'logs') if log_path is not None else os.path.join(os.getcwd(),
                                                                                        'logs')

    # Validate the path and clear the existing log file
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    normal_file = os.path.join(log_path, f'{name}.log')
    error_file = os.path.join(log_path, f'{name}.issues.log')
    with open(normal_file, 'w', encoding='utf-8') as f:
        f.write('')

    # The different message formats to use
    msg_format = logging.Formatter(fmt='%(message)s')
    verbose_format = logging.Formatter(fmt='[%(asctime)s - %(levelname)8s] %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    error_format = logging.Formatter(
        fmt='[%(asctime)s - %(levelname)8s - %(name)20s - %(funcName)20s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Create the file handler
    normal_file_handler = CompactFileHandler(normal_file, logging.DEBUG, verbose_format)
    error_file_handler = CompactFileHandler(error_file, logging.WARNING, error_format)

    # Setup the console handlers for normal and errors
    console_handler = TQDMLoggingHandler(logging.INFO if not debug else logging.DEBUG,
                                         fmt=msg_format if not verbose else verbose_format)
    error_handler = TQDMLoggingHandler(logging.WARNING, fmt=error_format)

    # Set the environment variable to the names of the logger for use in other parts of the program
    os.environ['LOGGER_NAME'] = name
    os.environ['ISSUE_LOGGER_NAME'] = f'{name}.issue'

    # Create and register the two loggers
    logger = logging.getLogger(name)
    logger.addHandler(console_handler)
    logger.addHandler(normal_file_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    issue_logger = logging.getLogger(f'{name}.issue')
    issue_logger.addHandler(error_handler)
    issue_logger.addHandler(normal_file_handler)
    issue_logger.addHandler(error_file_handler)
    issue_logger.setLevel(logging.WARNING)
    issue_logger.propagate = False

    return logger, issue_logger
