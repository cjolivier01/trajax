import os

# By default, keep JAX tests on CPU to avoid long GPU compilation times.
# Users can override by setting `JAX_PLATFORMS` explicitly (e.g. `gpu`).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

