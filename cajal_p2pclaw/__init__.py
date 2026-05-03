"""
CAJAL — Native integration for the P2PCLAW scientific intelligence model.

Easy inference, chat, paper generation, and server for CAJAL-4B-P2PCLAW.
"""

__version__ = "1.0.0"
__author__ = "Francisco Angulo de Lafuente (Agnuxo1)"
__license__ = "MIT"

from cajal_p2pclaw.model import CAJALModel, load_model
from cajal_p2pclaw.chat import CAJALChat, chat
from cajal_p2pclaw.generator import PaperGenerator
from cajal_p2pclaw.citations import find_references, format_reference
from cajal_p2pclaw.tribunal import Tribunal
from cajal_p2pclaw.formats import to_markdown, to_latex, to_text, save

__all__ = [
    "CAJALModel",
    "load_model",
    "CAJALChat",
    "chat",
    "PaperGenerator",
    "find_references",
    "format_reference",
    "Tribunal",
    "to_markdown",
    "to_latex",
    "to_text",
    "save",
    "__version__",
]
