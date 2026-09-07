"""Period refinement: recover the period a model prefers, after classification.

Feature generation reports the period at which a light curve repeats.  For an
eclipsing binary that is half the orbital period, and every period-finding
algorithm agrees on the halved value, so no amount of cross-algorithm agreement
detects the error.  Recovering the orbital period requires fitting a model, and
the model has to match the class.

See registry.py for which classes are handled and why the rest are not.
"""

from .registry import REGISTRY, Model, lookup, supported

__all__ = ["REGISTRY", "Model", "lookup", "supported"]
