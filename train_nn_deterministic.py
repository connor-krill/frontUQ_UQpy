import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import UQpy.scientific_machine_learning as sml

# import logging  # Optional, display UQpy logs
# logger = logging.getLogger("UQpy")
# logger.setLevel(logging.INFO)

plt.style.use(["ggplot", "surg.mplstyle"])


class AerodynamicDataset(torch.utils.data.Dataset):
    def __init__(self, filename: str):
        """

        :param filename:
        """
        df = pd.read_csv(filename, index_col=0)
        input_columns = df.columns[0:5]
        output_columns = df.columns[5:]
        self.n = len(df)
        self.x = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float)
        self.y = torch.tensor(df[output_columns].to_numpy(), dtype=torch.float)

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# define the model
in_features = 5
width = 10
out_features = 4
network = nn.Sequential(
    nn.Linear(in_features, width),
    nn.Sigmoid(),
    nn.Linear(width, width),
    nn.Sigmoid(),
    nn.Linear(width, out_features),
)
model = sml.FeedForwardNeuralNetwork(network)

# define the dataset
filename = "./data/aerodynamics_data_100.csv"
dataset = AerodynamicDataset(filename)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.5, 0.5])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset)

# define and run the training scheme
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
trainer = sml.Trainer(model, optimizer, scheduler=scheduler)
trainer.run(train_data=train_dataloader, test_data=test_data_loader, epochs=100)  # ToDo: why is the training so wrong

# compute deterministic NN predictions
model.eval()
x_test = test_dataset.dataset.x[test_dataset.indices]
nn_prediction = model(x_test)
nn_prediction = nn_prediction.detach().numpy()

# plot neural network and XFOIL results
fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True)
x_test = test_dataset.dataset.x[test_dataset.indices]
y_test = test_dataset.dataset.y[test_dataset.indices]

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
    nn_prediction[:, 0],
    label="NN Lift",
    color="black",
    marker="+",
    s=6**2,
)
ax0.set_title("NN Predictions for Lift")
ax0.set(xlabel="Angle of Attack", ylabel="Lift (CL)")
ax0.legend()

ax1.scatter(
    x_test[:, 0],
    y_test[:, 3],
    label="XFoil Torque",
    color="tab:gray",
    marker="o",
    s=10**2,
    alpha=0.5,
)
ax1.scatter(
    x_test[:, 0],
    nn_prediction[:, 3],
    label="NN Torque",
    color="black",
    marker="+",
    s=6**2,
)
ax1.set_title("NN Predictions for Torque")
ax1.set(xlabel="Angle of Attack", ylabel="Torque (CM)")
ax1.legend(loc="lower right")
fig.tight_layout()

# plot training and testing loss
fig, ax = plt.subplots()
ax.semilogy(trainer.history["train_loss"], label="Training Loss")
ax.semilogy(trainer.history["test_loss"], label="Test Loss")
ax.set_title("Training History of Deterministic NN", loc="left")
ax.set(xlabel="Epoch", ylabel="Log of Loss")
ax.legend()

plt.show()
