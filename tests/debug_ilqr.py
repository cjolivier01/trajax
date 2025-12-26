"""Debug iLQR to find the issue."""

import numpy as np
import torch
import sys
sys.path.insert(0, '/mnt/home/colivier-local/src/bare_trajax')
from trajax_torch import optimizers as torch_optimizers

# Simple LQR problem
def cost_torch(x, u, t):
    Q = 1.0
    R = 0.1
    return 0.5 * Q * torch.sum(x**2) + 0.5 * R * torch.sum(u**2)

def dynamics_torch(x, u, t):
    A = 1.1
    B = 0.5
    return A * x + B * u

T = 20
x0_torch = torch.tensor([1.0], dtype=torch.float64)
U_torch = torch.zeros((T, 1), dtype=torch.float64)

print("Initial objective:", torch_optimizers.objective(cost_torch, dynamics_torch, U_torch, x0_torch))

# Test gradient computation
X = torch_optimizers._rollout(dynamics_torch, U_torch, x0_torch)
timesteps = torch.arange(X.shape[0])

cost_gradients = torch_optimizers.linearize(cost_torch)
q, r = cost_gradients(X, torch_optimizers.pad(U_torch), timesteps)

dynamics_jacobians = torch_optimizers.linearize(dynamics_torch)
A, B = dynamics_jacobians(X, torch_optimizers.pad(U_torch), timesteps)

print(f"q shape: {q.shape}, sample values: {q[:3]}")
print(f"r shape: {r.shape}, sample values: {r[:3]}")
print(f"A shape: {A.shape}, sample: {A[0]}")
print(f"B shape: {B.shape}, sample: {B[0]}")

gradient, adjoints, final_p = torch_optimizers.adjoint(A, B, q, r)
print(f"Computed gradient shape: {gradient.shape}, sample: {gradient[:3]}")
print(f"Gradient norm from adjoint: {torch.linalg.norm(gradient)}")

# Manually run one iteration to debug
T = 20
n = 1
device = x0_torch.device
dtype = x0_torch.dtype

quadratizer = torch_optimizers.quadratize(cost_torch)
Q, R, M = quadratizer(X, torch_optimizers.pad(U_torch), timesteps)

print(f"\nQ shape: {Q.shape}, sample: {Q[0]}")
print(f"R shape: {R.shape}, sample: {R[0]}")
print(f"M shape: {M.shape}, sample: {M[0]}")

c = torch.zeros((T, n), device=device, dtype=dtype)
K, k, P, p = torch_optimizers.tvlqr(Q, q, R, r, M, A, B, c)

print(f"K shape: {K.shape}, sample: {K[0]}")
print(f"k shape: {k.shape}, sample: {k[0]}")

# Try a rollout with these gains
X_new, U_new = torch_optimizers.ddp_rollout(dynamics_torch, X, U_torch, K, k, 1.0)
print(f"X_new[0]: {X_new[0]}, X[0]: {X[0]}")
print(f"U_new[0]: {U_new[0]}, U_torch[0]: {U_torch[0]}")
print(f"X_new[-1]: {X_new[-1]}, X[-1]: {X[-1]}")

obj_new = torch_optimizers.objective(cost_torch, dynamics_torch, U_new, x0_torch)
print(f"New objective: {obj_new}, Old objective: {torch_optimizers.objective(cost_torch, dynamics_torch, U_torch, x0_torch)}")

# Now test line search
obj_old = torch_optimizers.objective(cost_torch, dynamics_torch, U_torch, x0_torch)
X_ls, U_ls, obj_ls, alpha_ls = torch_optimizers.line_search_ddp(
    cost_torch, dynamics_torch, X, U_torch, K, k, obj_old
)
print(f"\nLine search results:")
print(f"  Returned objective: {obj_ls}")
print(f"  Returned alpha: {alpha_ls}")
print(f"  U_ls[0]: {U_ls[0]}")
print(f"  Did it improve? {obj_ls < obj_old}")
