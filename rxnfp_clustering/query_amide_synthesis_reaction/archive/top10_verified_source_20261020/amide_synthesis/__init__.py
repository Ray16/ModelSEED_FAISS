"""Reaction search → enzyme mapping → selection → structural modeling → evaluation.

Imports never execute a stage or retrieve remote data. Old flat module names
resolve lazily to their canonical implementations for frozen-run compatibility.
"""
from ._compat import install as _install_compatibility
_install_compatibility()
