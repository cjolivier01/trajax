"""Simple helpers for converting continuous-time dynamics to discrete-time using PyTorch."""

import torch


def euler(dynamics, dt=0.01):
  """First-order Euler integration."""
  def integrator(x, u, t):
    return x + dt * dynamics(x, u, t)
  return integrator


def rk4(dynamics, dt=0.01):
  """Fourth-order Runge-Kutta integration."""
  def integrator(x, u, t):
    dt2 = dt / 2.0
    k1 = dynamics(x, u, t)
    k2 = dynamics(x + dt2 * k1, u, t)
    k3 = dynamics(x + dt2 * k2, u, t)
    k4 = dynamics(x + dt * k3, u, t)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
  return integrator
