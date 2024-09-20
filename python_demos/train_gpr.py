import numpy as np
import UQpy as uq
import umbridge
np.random.seed(0)

import logging  # Optional, display UQpy logs
logger = logging.getLogger("UQpy")
logger.setLevel(logging.INFO)

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
# x = distribution.rvs(n)  # random samples
x = uq.LatinHypercubeSampling(distribution, n, random_state=0).samples  # LHS samples

# compute outputs using UM-Bridge
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
lift_train = outputs[0:split, 0]
torque_train = outputs[0:split, 3]
x_test = x[split:]
lift_test = outputs[split:, 0]
torque_test = outputs[split:, 3]

# define and train gaussian process regression
kernel = uq.utilities.RBF()
regression_model = uq.LinearRegression()
optimizer = uq.MinimizeOptimizer()
gpr_lift = uq.GaussianProcessRegression(
    kernel,
    [1] * 7,
    regression_model=regression_model,
    optimizer=optimizer,
    optimizations_number=10,
    noise=True,
)
gpr_torque = uq.GaussianProcessRegression(
    kernel,
    [1] * 7,
    regression_model=regression_model,
    optimizer=optimizer,
    optimizations_number=10,
    noise=True,
)
gpr_lift.fit(x_train, lift_train)
gpr_torque.fit(x_train, torque_train)

# compute GPR predictions
lift_prediction, lift_std = gpr_lift.predict(x_test, return_std=True)
torque_prediction, torque_std = gpr_torque.predict(x_test, return_std=True)

# plot results
import matplotlib.pyplot as plt

plt.style.use(["ggplot", "surg.mplstyle"])

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(7, 5))
ax0.scatter(
    x_test[:, 0],
    lift_test,
    label="XFoil Lift",
    color="darkgray",
    marker="o",
    s=10**2,
)
ax0.errorbar(
    x_test[:, 0],
    lift_prediction[:, 0],
    yerr=3 * lift_std,
    label="GPR $\mu \pm 3\sigma$",
    color="tab:blue",
    ecolor="lightsteelblue",
    marker="D",
    linestyle="none",
)
ax0.set_title("GPR Predictions of Lift")
ax0.set(xlabel="Angle of Attack", ylabel="Lift (CL)")
ax0.legend(loc="upper left", framealpha=1.0)

ax1.scatter(
    x_test[:, 4],
    torque_test,
    label="XFoil Torque",
    color="darkgray",
    marker="o",
    s=10**2,
)
ax1.errorbar(
    x_test[:, 4],
    torque_prediction[:, 0],
    yerr=3 * torque_std,
    label="GPR $\mu \pm 3\sigma$",
    color="tab:blue",
    ecolor="lightsteelblue",
    marker="D",
    linestyle="none",
)
ax1.set_title("GPR Predictions for Torque")
ax1.set(xlabel="Flap Deflection", ylabel="Torque (CM)")
ax1.legend(loc="upper left", framealpha=1.0)
fig.tight_layout()

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(7, 5))
ax0.errorbar(
    lift_test,
    lift_prediction[:, 0],
    yerr=3 * lift_std,
    color="tab:blue",
    marker="o",
    alpha=0.4,
    linestyle="none",
)
ax1.errorbar(
    torque_test,
    torque_prediction[:, 0],
    yerr=3 * torque_std,
    color="tab:blue",
    marker="o",
    alpha=0.4,
    linestyle="none",
)
# format the plots
ax0.set_title("Comparison of Lift")
ax0.set(xlabel="XFoil Lift", ylabel="GPR Lift")
ax1.set_title("Comparison of Torque")
ax1.set(xlabel="XFoil Torque", ylabel="GPRTorque")
ax0.set_yticks(ax0.get_xticks())
for ax in (ax0, ax1):
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, color="white", linewidth=0.5, zorder=1)
    ax.set_aspect("equal")
fig.tight_layout()
fig.suptitle("Gaussian Process Regression Surrogate")
plt.show()
