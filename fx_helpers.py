"""Backward-compatibility shim for Rust PyO3 imports.

The Rust code does py.import_bound("fx_helpers") which needs this module
on sys.path. Redirects to the actual implementation in retab._helpers.
"""
from retab._helpers import *  # noqa: F401,F403
