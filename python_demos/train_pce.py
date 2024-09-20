import numpy as np
import UQpy as uq
import umbridge

np.random.seed(0)

import logging  # Optional, display UQpy logs

logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)

# generate training data
# define inputs using UQpy
marginals = [
    uq.Normal(0.0, 0.1),
    uq.Normal(500_000, 2_500),
    uq.Normal(0.3, 0.015),
    uq.Normal(0.7, 0.021),
    uq.Normal(0, 0.08),
]
distribution = uq.JointIndependent(marginals)
n = 100
x = distribution.rvs(n)  # random samples
# x = uq.LatinHypercubeSampling(distribution, n, random_state=0).samples  # LHS samples

# define outputs using UM-Bridge
model = umbridge.HTTPModel("http://localhost:53687", "forward")
outputs = np.zeros((n, 4))
for i in range(n):
    if i % (n // 10) == 9:
        print(f"UM-Bridge Model Evaluations: {i + 1} / {n}")
    model_input = [x[i].tolist()]  # model input is a list of lists
    outputs[i] = model(model_input)[0]

# split into training and testing data sets
split = int(n * 0.5)
x_train = x[0:split]
y_train = outputs[0:split]
x_test = x[split:]
y_test = outputs[split:]

# define and fit Polynomial Chaos Expansion
polynomial_basis = uq.TotalDegreeBasis(distribution, max_degree=2)
regression_method = uq.LeastSquareRegression()
pce = uq.PolynomialChaosExpansion(polynomial_basis, regression_method)
pce.fit(x_train, y_train)

# compute PCE predictions, moments, and Sobol indices
pce_prediction = pce.predict(x_test)
pce_mean, pce_variance = pce.get_moments()
pce_sensitivity = uq.PceSensitivity(pce)
pce_sensitivity.run()

# compare moments from PCE and Monte Carlo
pce_prediction = pce.predict(x_test)

pce_mean, pce_variance = pce.get_moments()
print("PCE Statistics", "Lift".ljust(10), "Torque")
print("Mean".ljust(14), f"{pce_mean[0]:5.4f} {pce_mean[3]:11.4f}")
print("Variance".ljust(14), f"{pce_variance[0]:.4e} {pce_variance[3]:.4e}")
print()

pce_sensitivity = uq.PceSensitivity(pce)
pce_sensitivity.run()

input_names = [
    "Angle of Attack",
    "Reynolds Number",
    "Upper Surface Trip Location",
    "Lower Surface Trip Location",
    "Flap Deflection"
]
print("PCE Sobol Indices".ljust(27), "Lift".ljust(6), "Torque")
for i in range(len(input_names)):
    sobol_lift = pce_sensitivity.first_order_indices[i, 0]
    sobol_torque = pce_sensitivity.first_order_indices[i, 3]
    print(input_names[i].ljust(27), f"{sobol_lift: .2f} {sobol_torque: 6.2f}")

# plot predictions and results
import matplotlib.pyplot as plt

plt.style.use(["ggplot", "surg.mplstyle"])

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(7, 5))
ax0.scatter(
    x_test[:, 0],
    y_test[:, 0],
    label="XFoil Lift",
    color="darkgray",
    marker="o",
    s=10**2,
)
ax0.scatter(
    x_test[:, 0],
    pce_prediction[:, 0],
    label="PCE Lift",
    color="tab:blue",
    marker="D",
    s=6**2,
)
ax0.set_title("PCE Predictions for Lift")
ax0.set(xlabel="Angle of Attack", ylabel="Lift (CL)")
ax0.legend(loc="upper left", framealpha=1.0)

ax1.scatter(
    x_test[:, 4],
    y_test[:, 3],
    label="XFoil Torque",
    color="darkgray",
    marker="o",
    s=10**2,
)
ax1.scatter(
    x_test[:, 4],
    pce_prediction[:, 3],
    label="PCE Torque",
    color="tab:blue",
    marker="D",
    s=6**2,
)
ax1.set_title("PCE Predictions for Torque")
ax1.set(xlabel="Flap Deflection", ylabel="Torque (CM)")
ax1.legend(loc="upper left", framealpha=1.0)
fig.suptitle("Polynomial Chaos Expansion Surrogate")
fig.tight_layout()

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(7, 4))
ax0.scatter(y_test[:, 0], pce_prediction[:, 0], color="tab:blue", alpha=0.4, zorder=2)
ax1.scatter(y_test[:, 3], pce_prediction[:, 3], color="tab:blue", alpha=0.4, zorder=2)

# format the plots
ax0.set_title("Comparison of Lift")
ax0.set(xlabel="XFoil Lift", ylabel="PCE Lift")
ax1.set_title("Comparison of Torque")
ax1.set(xlabel="XFoil Torque", ylabel="PCE Torque")
ax1.set_yticks(ax1.get_xticks())
for ax in (ax0, ax1):
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, color="white", linewidth=0.5, zorder=1)
    ax.set_aspect("equal")
    ax.set(xlim=xlim, ylim=xlim)
fig.suptitle("Polynomial Chaos Expansion Surrogate")
fig.tight_layout()


plt.show()
