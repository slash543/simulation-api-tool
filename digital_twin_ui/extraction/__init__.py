"""
Extraction module.

Exposes the main parsing interfaces:
  - XpltParser / XpltData / extract_contact_pressure  (legacy xplt_parser.py)
  - SimulationCase                                      (xplt_core.py — full pipeline)
"""
from .xplt_parser import XpltData, XpltParser, extract_contact_pressure, parse_xplt
from .xplt_core import SimulationCase

__all__ = [
    "XpltParser",
    "XpltData",
    "extract_contact_pressure",
    "parse_xplt",
    "SimulationCase",
]
