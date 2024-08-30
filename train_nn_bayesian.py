import logging
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import UQpy.scientific_machine_learning as sml

# import logging  # Optional, display UQpy logs
# logger = logging.getLogger("UQpy")
# logger.setLevel(logging.INFO)

plt.style.use(["ggplot", "surg.mplstyle"])


class AerodynamicDataset(Dataset):

    def __init__(self, filename: str):
        """

        :param filename: Name of .csv containing 5 columns of inputs followed by 4 columns of outputs where each row is a sample.
        """
        self.filename: str = filename
        df = pd.read_csv(self.filename, index_col=0)
        self.n: int = len(df)
        x_columns = df.columns[0:5]
        y_columns = df.columns[5:9]
        self.x = torch.tensor(df[x_columns].to_numpy(), dtype=torch.float)
        self.y = torch.tensor(df[y_columns].to_numpy(), dtype=torch.float)

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# define data
filename = "./data/aerodynamics_data_1000.csv"
dataset = AerodynamicDataset(filename)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset)

# define model
in_features = 5
width = 8
out_features = 4
network = nn.Sequential(
    sml.BayesianLinear(in_features, width),
    nn.ReLU(),
    sml.BayesianLinear(width, width),
    nn.ReLU(),
    sml.BayesianLinear(width, out_features),
)
model = sml.FeedForwardNeuralNetwork(network)

# define and run optimization
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
bbb_trainer = sml.BBBTrainer(model, optimizer, scheduler=scheduler)
bbb_trainer.run(train_data=train_dataloader, test_data=test_dataloader, epochs=500, num_samples=10)

# compute Bayesian NN predictions
x_test = test_dataset.dataset.x[test_dataset.indices]
y_test = test_dataset.dataset.y[test_dataset.indices]
bnn_predictions = model(x_test)
bnn_predictions = bnn_predictions.detach().numpy()

# plot predictions and xfoil results
fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True)
ax0.scatter(x_test[:, 0], y_test[:, 0], label="XFoil Lift")
ax0.scatter(x_test[:, 0], bnn_predictions[:, 0], label="BNN Lift")
ax0.set_title("Bayesian NN Predictions for Lift")
ax0.set(xlabel="Angle of Attack", ylabel="Lift (CL)")
ax0.legend(loc="lower right")

# plot results
fig, ax = plt.subplots()
ax.semilogy(bbb_trainer.history["train_loss"], label="Training Loss")
ax.semilogy(bbb_trainer.history["test_nll"], label="Test Loss")
ax.set_title("Training History", loc="left")
ax.set(xlabel="Epoch")
ax.legend()

plt.show()
