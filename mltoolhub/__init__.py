try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata 

try:
    __version__ = metadata.version(__package__)

except metadata.PackageNotFoundError:
    __version__ = '0.0.0.dev'


# Basic utils 
from mltoolhub.basics import get_quick_summary, get_summary_plots
import mltoolhub.prologger as prologger

__all__ = ['get_quick_summary', 'get_summary_plots', 'prologger', '__version__']
