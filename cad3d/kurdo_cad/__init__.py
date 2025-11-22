"""
KURDO CAD - Professional CAD/BIM Design System
A comprehensive design platform integrating AutoCAD, Revit, and Civil 3D capabilities.
"""

__version__ = "1.0.0"
__author__ = "KURDO AI"

from .core_engine import KurdoCADEngine
from .drawing_tools import DrawingToolkit
from .bim_tools import BIMToolkit
from .civil_tools import CivilToolkit

__all__ = [
    'KurdoCADEngine',
    'DrawingToolkit',
    'BIMToolkit',
    'CivilToolkit'
]
