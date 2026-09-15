"""elxai — electricity load forecasting under normal and extreme weather, with XAI.

The package is organised as a one-way layered pipeline:

    data  ->  models  ->  training  ->  xai  ->  reporting

Each layer only depends on the ones to its left. `experiments` sits on top and
wires the layers together from a YAML config. The scientific contract that the
rest of the package relies on lives in :mod:`elxai.data.schema`.
"""

from elxai.version import __version__

__all__ = ["__version__"]
