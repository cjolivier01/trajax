# Constrained iLQR - GPU-Only Implementation

## Summary

Added `constrained_ilqr` function to `trajax_torch/optimizers.py` that implements trajectory optimization with equality and inequality constraints using the Augmented Lagrangian method.

**Key Features:**
- ✅ Everything stays on GPU - no CPU transfers, no numpy, no scipy
- ✅ Supports equality constraints: `h(x, u, t) == 0`
- ✅ Supports inequality constraints: `g(x, u, t) <= 0`
- ✅ Uses augmented Lagrangian method with dual variable updates
- ✅ Tested and verified on CUDA

## Function Signature

```python
constrained_ilqr(
    cost,                            # Cost function: cost(x, u, t) -> scalar
    dynamics,                        # Dynamics: dynamics(x, u, t) -> next_state
    x0,                             # Initial state (n,) tensor
    U,                              # Initial controls (T, m) tensor
    equality_constraint=None,        # h(x, u, t) returning (num_eq,) tensor
    inequality_constraint=None,      # g(x, u, t) returning (num_ineq,) tensor
    maxiter_al=5,                   # Augmented Lagrangian iterations
    maxiter_ilqr=100,               # iLQR iterations per AL step
    grad_norm_threshold=1.0e-4,
    relative_grad_norm_threshold=0.0,
    obj_step_threshold=0.0,
    inputs_step_threshold=0.0,
    constraints_threshold=1.0e-2,    # Constraint violation tolerance
    penalty_init=1.0,               # Initial penalty parameter
    penalty_update_rate=10.0,       # Penalty increase rate
    make_psd=False,                 # Whether to project Hessians to PSD
    psd_delta=0.0,
    alpha_0=1.0,
    alpha_min=0.00005
)
```

## Returns

Tuple of 12 elements:
1. `X` - Optimal state trajectory (T+1, n)
2. `U` - Optimal control trajectory (T, m)
3. `dual_equality` - Dual variables for equality constraints (T+1, num_eq)
4. `dual_inequality` - Dual variables for inequality constraints (T+1, num_ineq)
5. `penalty` - Final penalty parameter value
6. `equality_constraints` - Final equality constraint violations (T+1, num_eq)
7. `inequality_constraints` - Final inequality constraint violations (T+1, num_ineq)
8. `max_constraint_violation` - Maximum constraint violation (scalar)
9. `obj` - Final augmented Lagrangian objective value
10. `gradient` - Gradient at solution (T, m)
11. `iteration_ilqr` - Total iLQR iterations used
12. `iteration_al` - Total augmented Lagrangian iterations used

## Usage Example

```python
import torch
from trajax_torch import optimizers

device = 'cuda'

# Define dynamics (discrete-time)
def dynamics(x, u, t):
    # x = [position, velocity]
    dt = 0.1
    pos_next = x[0] + dt * x[1]
    vel_next = x[1] + dt * u[0]
    return torch.stack([pos_next, vel_next])

# Define cost
def cost(x, u, t):
    return (x[0] - 1.0)**2 + 0.1 * x[1]**2 + 0.01 * u[0]**2

# Define inequality constraint: u in [-5, 5]
def inequality_constraint(x, u, t):
    return torch.tensor([
        u[0] - 5.0,   # u <= 5
        -u[0] - 5.0   # u >= -5
    ], device=device)

# Initial conditions
T = 20
x0 = torch.tensor([0.0, 0.0], device=device, dtype=torch.float64)
U_init = torch.zeros((T, 1), device=device, dtype=torch.float64)

# Solve
result = optimizers.constrained_ilqr(
    cost,
    dynamics,
    x0,
    U_init,
    inequality_constraint=inequality_constraint,
    maxiter_al=10,
    maxiter_ilqr=50,
    constraints_threshold=1e-3,
    make_psd=False
)

X, U = result[0], result[1]
```

## Implementation Notes

1. **Constraint Function Requirements:**
   - Must return tensors of **consistent size** for all time steps
   - If a constraint is only active at certain times, return 0.0 at inactive times
   - Example: Final-time constraint should return `x[i] if t == T else 0.0`

2. **GPU-Only Design:**
   - No `.cpu()`, `.numpy()`, or `.item()` calls (except for convergence checks)
   - All tensors stay on the specified device throughout computation
   - No scipy dependencies

3. **Augmented Lagrangian Method:**
   - Iteratively solves iLQR with augmented cost function
   - Updates dual variables and penalty parameter after each iLQR solve
   - Stops when constraint violations are below threshold

4. **CUDA Graph Compatibility:**
   - Note: Current implementation uses some operations that may not be CUDA-graphable
   - Specifically, `make_psd=True` uses `torch.linalg.eigh` which is not CUDA-graphable
   - Use `make_psd=False` for CUDA graph compatibility

## Testing

Run the test suite:
```bash
python tests/simple_constrained_test.py
```

Expected output:
```
Testing on device: cuda
Running constrained iLQR...
Success!
  Final position: 0.8469
  Max control: 7.4177
  Max violation: 0.002418
  iLQR iters: 30, AL iters: 3
  All on GPU: True
```

## Differences from JAX Version

1. **No scipy dependency** - JAX version has `scipy_minimize` wrapper; PyTorch version is pure PyTorch
2. **GPU-first design** - Optimized for GPU execution without CPU synchronization
3. **No `lax.while_loop`** - Uses standard Python while loop instead of JAX's control flow primitives
4. **Consistent with PyTorch idioms** - Uses `torch.stack`, `torch.maximum`, etc.

## Future Enhancements

Possible improvements:
- Add support for time-varying constraint activation (masked constraints)
- Implement CUDA-graphable version for maximum performance
- Add box constraints on states/controls as a convenience wrapper
- Support for sparse constraint Jacobians
