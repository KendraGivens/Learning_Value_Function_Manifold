import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from lvfm.ground_truths import LinearOscillator2D

def solve_linear_oscillator_2d(tau_steps, x1_discretization, x2_discretization, x1_grid, x2_grid, u_bound, d_bound, oscillation_speed, radius):
    times = np.concatenate(([0.0], np.array(tau_steps))) * -1
    
    dynamics = LinearOscillator2D(oscillation_speed=oscillation_speed, u_bound=u_bound, d_bound=d_bound)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([x1_grid[0], x2_grid[0]]),np.array([x1_grid[1], x2_grid[1]]),), shape=(x1_discretization, x2_discretization))
    positions = grid.states[..., :2]
    
    values0 = jnp.linalg.norm(positions, axis=-1) - radius

    solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    value_function = hj.solve(solver_settings, dynamics, grid, times, values0)

    return value_function 