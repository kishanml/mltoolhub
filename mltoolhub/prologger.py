import os
import sys
import inspect
import traceback

import logging
from enum import Enum
import datetime as dt
from typing import Literal, List, Union, Callable, Any


# -----------------------------------PROLOGGER-------------------------------------



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
    handler_type : Literal["file", "rotate", None] = None,
    rotation_handler_config : List[Union[float,float]] = [2**13,2**13]):

    global __logger__

    try:
        if handler_type is not None:
            log_dir = os.path.dirname(log_filepath)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

        if __logger__ is None:

            __logger__ = logging.getLogger('root')
            __logger__.setLevel(logging.DEBUG) 

            if handler_type is not None:

                if  handler_type == "file":
                    fh = logging.FileHandler(log_filepath,mode='a')
                    fh.setLevel(logging.DEBUG)

                elif handler_type == "rotate":
                    fh = logging.handlers.RotatingFileHandler(log_filepath,*rotation_handler_config)
                    fh.setLevel(logging.DEBUG)

                else:
                    raise ValueError('handler_type should be file or rotate.')
                
                __logger__.addHandler(fh)
                
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(LevelBasedFormatter(log_format))

            __logger__.addHandler(ch)
            
    
    except Exception as exc : 
        raise Exception(f'Error occured while configuring Logger : {exc}\n{traceback.format_exc()}')
    



## logger functions 
##- Resolves the issue of the decorator path appearing as the error location in standard logging.

def _caller_info_() -> str : 

    __is_ipykernel__ : bool = 'ipykernel' in sys.modules or 'IPython' in sys.modules
    if __is_ipykernel__:

        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell is not None:

                cell_count = shell.execution_count
                return f"[Notebook : #{cell_count}] "
        except ImportError:
            return "[Notebook] "
        

    try: 
        cur_frame = inspect.stack()[-1]
        return f"[{cur_frame.filename} : #{str(cur_frame.lineno)}] "
    
    except:
        return "[Unknown Caller] "
    
def _log_msg_(log_func : Callable[[str],None], *message : Any) -> None:

    global __logger__

    try:

        inspect_info = _caller_info_()

        messages = list(message)
        messages[0] = inspect_info + str(messages[0])
        final_message = " ".join(str(msg) for msg in messages)

        if __logger__ is not None:
            log_func(final_message)

    
    except Exception as exc:

        print(f'Error occured in _log_msg_ : {exc}\n{traceback.format_exc()}',file=sys.stderr)


# wrapper functions

def log_info(*message: Any) -> None:
    """Logs an INFO level message."""
    global __logger__
    if __logger__ is not None:
        _log_msg_(__logger__.info, *message)

def log_warn(*message: Any) -> None:
    """Logs a WARNING level message."""
    global __logger__
    if __logger__ is not None:
        _log_msg_(__logger__.warning, *message)

def log_debug(*message: Any) -> None:
    """Logs a DEBUG level message."""
    global __logger__
    if __logger__ is not None:
        _log_msg_(__logger__.debug, *message)

def log_error(*message: Any) -> None:
    """Logs a ERROR level message."""
    global __logger__
    if __logger__ is not None:
        _log_msg_(__logger__.error, *message)

def log_critical(*message: Any) -> None:
    """Logs a CRITICAL level message."""
    global __logger__
    if __logger__ is not None:
        _log_msg_(__logger__.critical, *message)



## trace all decorator

def trace_errors():

    def execute(func):

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                if __logger__ is not None:
                    _log_msg_(__logger__.error,"Traceback:\n" + traceback.format_exc())

        return wrapper

    return execute



