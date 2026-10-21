"""Bootstrap pinned core without adopting its deployment environment defaults."""

import importlib
import os


def load_core_config():
    # harness2.config loads .env and sets Vertex defaults at import time. The
    # subscription adapter uses core scheduling only, never its provider routes.
    # Bootstrap once, before core threads; preserve the caller's exact environment.
    before = dict(os.environ)
    try:
        return importlib.import_module("harness2.config")
    finally:
        for key in set(os.environ) - set(before):
            del os.environ[key]
        os.environ.update(before)
