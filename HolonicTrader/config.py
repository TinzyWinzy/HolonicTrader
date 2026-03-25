"""
HolonicTrader Configuration Passthrough
Consolidated into root config.py (2026-03-09)
"""
import os
import sys

# Import everything from the root config.py
_root_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.py'))

import importlib.util
_spec = importlib.util.spec_from_file_location("root_config", _root_config_path)
_root_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_config)

# Re-export all root config attributes to this module's globals
for _attr in dir(_root_config):
    if not _attr.startswith('_'):
        globals()[_attr] = getattr(_root_config, _attr)
