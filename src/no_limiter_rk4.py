# solve 1D advection using schemes of varying order, plot to compare
import numpy as np
import math
import matplotlib.pyplot as plt
from util.solve import AdvectionSolver


# inputs
ic_type = "square"  # initial condition type
orders = range(1, 6)  # solve using schemes of these orders
a = 1  # tranpsort speed
x_bounds = [0, 1]  # spatial domain
h = 0.02  # grid size
T = 2  # solving time
Dt = 0.8 * h / a  # time step size

# array of x-values
x_interface = np.arange(x_bounds[0], x_bounds[1] + h, h)
x = 0.5 * (x_interface[:-1] + x_interface[1:])  # x at cell centers

# array of t-values
t = np.arange(0, T + Dt, Dt)

# initial values of u
if ic_type == "sinus":
    u0 = np.cos(2 * math.pi * x)
elif ic_type == "square":
    u0 = np.array(
        [np.heaviside(i - 0.25, 1) - np.heaviside(i - 0.75, 1) for i in x]
    )

# advect for T orbits and the initial state should match the final state
# do this for many orders
plt.plot(x, u0, label="t = 0")
for order in orders:
    advection_solution = AdvectionSolver(x0=u0, t=t, h=h, a=a, order=order)
    advection_solution.rk4()
    u = advection_solution.x
    plt.plot(
        x, u[:, -1], "--", marker="o", mfc="none", label=f"order {order} + rk4"
    )
plt.title(f"{T} orbits of constant 1D advection" " within a periodic box")
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.legend()
plt.show()
