import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import UQpy as uq

# import logging  # Optional, display UQpy logs
# logger = logging.getLogger("UQpy")
# logger.setLevel(logging.INFO)

plt.style.use(["ggplot", "surg.mplstyle"])


# load data
filename = "./data/aerodynamics_data_100.csv"
df = pd.read_csv(filename, index_col=0)
input_columns = df.columns[0:5]
output_columns = df.columns[5:9]
n_samples = len(df)

# split into training and testing data sets
split = int(n_samples * 0.5)
x = df[input_columns].to_numpy()
y = df[output_columns].to_numpy()
x_train = x[0:split]
y_train = y[0:split]
x_test = x[split:]
y_test = y[split:]

# define and train gaussian process regression
kernel = uq.utilities.RBF()
optimizer = uq.MinimizeOptimizer()
gpr = uq.GaussianProcessRegression(
    kernel,
    [1] * 7,
    regression_model=uq.LinearRegression(),
    optimizer=optimizer,
    optimizations_number=10,
    noise=True,
)
gpr.fit(x_train, y_train)

# compute GPR predictions
gp_prediction, gp_std = gpr.predict(x_test, return_std=True)


# plot results
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
ax0.scatter(
    x_test[:, 0],
    y_test[:, 0],
    label="XFoil Lift",
    color="tab:gray",
    marker="o",
    s=10**2,
    alpha=0.5,
)
ax0.scatter(
    x_test[:, 0],
    gp_prediction[:, 0],
    label="PCE Lift",
    color="black",
    marker="+",
    s=6**2,
)
# plot histogram of error
mse_error = np.linalg.norm(y_pred - y_test, axis=1)
fig, ax = plt.subplots()
ax.hist(mse_error, bins=30, edgecolor=ax.get_facecolor())
ax.set_title("Histogram of Test Data Error")
ax.set(xlabel="MSE $|| y - \hat{y} ||$", ylabel="Log of Counts")
ax.set_yscale("log")



# plot results
fig, ax_array = plt.subplots(nrows=2, ncols=2)
for i in range(4):
    ax = ax_array.flatten()[i]
    ax.scatter(y_test[:, i], y_pred[:, i], s=4**2, alpha=0.2, zorder=2)

    # x = np.arange(n_samples - split)
    # ax.scatter(x, y_test[:, i], label="Test Data")
    # ax.scatter(x, y_pred[:, i], label="GPR Predictions")
    # ax.scatter(x, abs(y_pred[:, i] - y_test[:, i]), label="Error $|y - \hat{y}|$")

    # format the plots
    ax.set_title(f"{output_columns[i]}", fontsize="medium")
    ax.set(xlabel="Test", ylabel="Prediction")
    ax.set_aspect("equal")
    ax.ticklabel_format(
        axis="both", style="scientific", useMathText=True, scilimits=(1, 2)
    )
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, ylim, color="white", linewidth=0.5, zorder=1)
    ax.set(xlim=xlim, ylim=ylim)
fig.suptitle("Gaussian Process Regression Surrogate")
fig.subplots_adjust(hspace=0.6)
plt.show()
