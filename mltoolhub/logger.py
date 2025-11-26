import os
import logging
import datetime as dt
from enum import Enum


class TERMCOLOR(Enum):
    RESET = "\x1b[0m"
    LIGHT_CYAN = "\x1b[1;36m"
    LIGHT_GREEN = "\x1b[1;32m"
    YELLOW = "\x1b[1;33m"
    LIGHT_RED = "\x1b[1;31m"
    LIGHT_PURPLE = "\x1b[1;35m"


LOG_FORMAT = "[%(asctime)s][%(name)s] %(levelname)s - %(message)s"

class LevelBasedFormatterForConsole(logging.Formatter):
    COLORS = {
        logging.DEBUG:    TERMCOLOR.LIGHT_CYAN.value,
        logging.INFO:     TERMCOLOR.LIGHT_GREEN.value,
        logging.WARNING:  TERMCOLOR.YELLOW.value,
        logging.ERROR:    TERMCOLOR.LIGHT_RED.value,
        logging.CRITICAL: TERMCOLOR.LIGHT_PURPLE.value,
    }

    def __init__(self, fmt=LOG_FORMAT, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        color = self.COLORS.get(record.levelno, TERMCOLOR.RESET.value)

        # format using your original LOG_FORMAT
        formatted = super().format(record)

        # wrap entire output in color
        return f"{color}{formatted}{TERMCOLOR.RESET.value}"



class proLogger:

    _configured = False
    _console_handler = None
    _file_handler = None

    @staticmethod
    def configure(
        log_filepath: str = f"logs/log_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        log_format: str = LOG_FORMAT
    ):

        if proLogger._configured:   
            return

        log_dir = os.path.dirname(log_filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(LevelBasedFormatterForConsole())

        # file handler
        fh = logging.FileHandler(log_filepath, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(log_format))

        # store on class
        proLogger._console_handler = ch
        proLogger._file_handler = fh
        proLogger._configured = True
        

    @staticmethod
    def get_logger(name: str):

        if not proLogger._configured:
            proLogger.configure()   

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:  
            logger.addHandler(proLogger._console_handler)
            logger.addHandler(proLogger._file_handler)

        logger.propagate = False
        return logger
    
    
