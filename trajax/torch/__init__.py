"""PyTorch backend for Trajax.

This subpackage provides GPU-first PyTorch implementations of Trajax's core
APIs (e.g. tvlqr and ilqr). The JAX backend remains available under the
top-level `trajax` modules.
"""

from . import integrators
from . import optimizers
from . import tvlqr

