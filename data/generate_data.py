import umbridge
import UQpy as uq
import numpy as np
import pandas as pd

model = umbridge.HTTPModel("http://localhost:54698", "forward")

marginals = [
    uq.Normal(0.0, 0.1),
    uq.Normal(500_000, 2_500),
    uq.Normal(0.3 * c, 0.015 * c),
    uq.Normal(0.7 * c, 0.021 * c),
    uq.Normal(0, 0.08),
]
distribution = uq.JointIndependent(marginals)
n = 100
samples = distribution.rvs(n)

outputs = np.zeros((n, 4))
for i in range(n):
    if i % (n // 10) == 0 or (i == n - 1):
        print(f"{i:,} / {n:,}")
    input = [samples[i].tolist()]  # model input is a list of lists
    outputs[i] = model(input)[0]

data = np.hstack((samples, outputs))
df = pd.DataFrame(
    data,
    columns=[
        "Angle of Attack",
        "Reynolds Number",
        "Upper Surface Trip Location",
        "Lower Surface Trip Location",
        "Flap Deflection",
        "Lift (CL)",
        "Total Resistance (CD)",
        "Resistance due to Pressure (CDp)",
        "Torque (CM)",
    ],
)
df.to_csv(f"aerodynamics_data_{n}.csv")
