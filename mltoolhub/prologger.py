import os
import inspect
import traceback

import logging
from enum import Enum
import datetime as dt
from typing import Literal, List, Union

# -------------------------------------------------------------------------


class TERMCOLOR(Enum):
    RESET = "\x1b[0m"
    LIGHT_CYAN = "\x1b[1;36m"
    LIGHT_GREEN = "\x1b[1;32m"
    YELLOW = "\x1b[1;33m"
    LIGHT_RED = "\x1b[1;31m"
    LIGHT_PURPLE = "\x1b[1;35m"


LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"


class LevelBasedFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: TERMCOLOR.LIGHT_CYAN.value,
        logging.INFO: TERMCOLOR.LIGHT_GREEN.value,
        logging.WARNING: TERMCOLOR.YELLOW.value,
        logging.ERROR: TERMCOLOR.LIGHT_RED.value,
        logging.CRITICAL: TERMCOLOR.LIGHT_PURPLE.value,
    }

    def __init__(self, fmt=LOG_FORMAT, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        color = self.COLORS.get(record.levelno, TERMCOLOR.RESET.value)
        formatted = super().format(record)
        return f"{color}{formatted}{TERMCOLOR.RESET.value}"


__logger__ : logging.Logger = None


def configure(
    log_filepath: str = f"logs/log_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    log_format: str = LOG_FORMAT,
    handler_type : Literal["file", "rotate"] = "file",
    rotation_handler_config : List[Union[float,float]] = [2**13,2**13]):

    global __logger__

    try:
        log_dir = os.path.dirname(log_filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if __logger__ is None:

            __logger__ = logging.getLogger('root')
            __logger__.setLevel(logging.DEBUG) 

            if handler_type == "file":
                fh = logging.FileHandler(log_filepath,mode='a')
                fh.setLevel(logging.DEBUG)

            elif handler_type == "rotate":
                fh = logging.handlers.RotatingFileHandler(log_filepath,*rotation_handler_config)
                fh.setLevel(logging.DEBUG)

            else:
                raise ValueError('handler_type should be file or rotate.')
            
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(LevelBasedFormatter(log_format))

            __logger__.addHandler(ch)
            __logger__.addHandler(fh)
    
    except Exception as exc : 
        raise Exception(f'Error occured while configuring Logger : {exc}\n{traceback.format_exc()}')
    

## logger functions
### Resolves the issue of the decorator path appearing as the error location in standard logging.

def log_info(*message) -> None:

    global __logger__

    try:
        cur_frame = inspect.stack()[-1]
        inspect_info = f"[{cur_frame.filename} : {str(cur_frame.lineno)}] "
        final_message = ' '.join([inspect_info + msg for msg in message])
        
        if __logger__ is not None:
            __logger__.info(final_message)
    
    except Exception as exc : 
        raise Exception(f'Error occured while executing log_info : {exc}\n{traceback.format_exc()}')


def log_debug(*message) -> None:

    global __logger__

    try:
        cur_frame = inspect.stack()[-1]
        inspect_info = f"[{cur_frame.filename} : {str(cur_frame.lineno)}] "
        final_message = ' '.join([inspect_info + msg for msg in message])
        
        if __logger__ is not None:
            __logger__.debug(final_message)
    
    except Exception as exc : 
        raise Exception(f'Error occured while executing log_debug : {exc}\n{traceback.format_exc()}')


def log_warn(*message) -> None:

    global __logger__

    try:
        cur_frame = inspect.stack()[-1]
        inspect_info = f"[{cur_frame.filename} : {str(cur_frame.lineno)}] "
        final_message = ' '.join([inspect_info + msg for msg in message])
        
        if __logger__ is not None:
            __logger__.warning(final_message)
    
    except Exception as exc : 
        raise Exception(f'Error occured while executing log_warn : {exc}\n{traceback.format_exc()}')



def log_error(*message) -> None:

    global __logger__

    try:
        cur_frame = inspect.stack()[-1]
        inspect_info = f"[{cur_frame.filename} : {str(cur_frame.lineno)}] "
        final_message = ' '.join([inspect_info + msg for msg in message])
        
        if __logger__ is not None:
            __logger__.error(final_message)
    
    except Exception as exc : 
        raise Exception(f'Error occured while executing log_error : {exc}\n{traceback.format_exc()}')



def log_critical(*message) -> None:

    global __logger__

    try:
        cur_frame = inspect.stack()[-1]
        inspect_info = f"[{cur_frame.filename} : {str(cur_frame.lineno)}] "
        final_message = ' '.join([inspect_info + msg for msg in message])
        
        if __logger__ is not None:
            __logger__.critical(final_message)
    
    except Exception as exc : 
        raise Exception(f'Error occured while executing log_critical : {exc}\n{traceback.format_exc()}')

        

## decorator

def log_exception(logger: logging.Logger):

    def execute(func):

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.log_error("Traceback:\n" + traceback.format_exc())

        return wrapper

    return execute



