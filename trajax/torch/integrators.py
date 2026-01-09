"""Torch-friendly helpers for converting continuous-time dynamics to discrete-time.

These helpers are backend-agnostic (they work with torch Tensors) and mirror
the API of `trajax.integrators`.
"""


def euler(dynamics, dt=0.01):
  return lambda x, u, t, *args: x + dt * dynamics(x, u, t, *args)


def rk4(dynamics, dt=0.01):
  def integrator(x, u, t, *args):
    dt2 = dt / 2.0
    k1 = dynamics(x, u, t, *args)
    k2 = dynamics(x + dt2 * k1, u, t, *args)
    k3 = dynamics(x + dt2 * k2, u, t, *args)
    k4 = dynamics(x + dt * k3, u, t, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

  return integrator

