"""Using Monte Carlo estimates to compute mean, variance, and Sobol indices of the XFoil model"""

import time
import logging
import numpy as np
import UQpy as uq
import umbridge

logger = logging.getLogger("UQpy")  # Optional, display UQpy logs
logger.setLevel(logging.INFO)

compute_statistics = False
compute_sobol = True

# define input distribution and XFOIL model
marginals = [
    uq.Normal(0.0, 0.1),
    uq.Normal(500_000, 2_500),
    uq.Normal(0.3, 0.015),
    uq.Normal(0.7, 0.021),
    uq.Normal(0, 0.08),
]
distribution = uq.JointIndependent(marginals)
uqpy_model = uq.RunModel(
    uq.PythonModel(
        model_script="model.py",
        model_object_name="xfoil",
    )
)

if compute_statistics:
    n = 10_000
    inputs = uq.LatinHypercubeSampling(distribution, n, random_state=0).samples
    model = umbridge.HTTPModel("http://localhost:53185", "forward")
    outputs = np.zeros((n, 4))
    for i in range(n):
        model_input = [inputs[i].tolist()]  # model input is a list of lists
        outputs[i] = model(model_input)[0]
    print("MC Statistics".ljust(13), "Lift".ljust(10), "Torque")
    print("Mean".ljust(13), f"{outputs[:, 0].mean():5.4f} {outputs[:, 3].mean():11.4f}")
    print("Variance".ljust(13), f"{outputs[:, 0].var():.4e} {outputs[:, 3].var():.4e}")

if compute_sobol:
    mc_sensitivity = uq.SobolSensitivity(uqpy_model, distribution, random_state=42)
    start = time.time()
    mc_sensitivity.run(n_samples=3_000)
    stop = time.time()
    mc_sensitivity_time = stop - start

    # print sobol indices
    input_names = [
        "Angle of Attack",
        "Reynolds Number",
        "Upper Surface Trip Location",
        "Lower Surface Trip Location",
        "Flap Deflection",
    ]
    print(f"Computed Sobol indices using MC in {mc_sensitivity_time:.4f} seconds")
    print("MC Sobol Indices".ljust(27), "Lift".ljust(7), "Torque")
    for i in range(len(input_names)):
        sobol_lift = mc_sensitivity.first_order_indices[i, 0]
        sobol_torque = mc_sensitivity.first_order_indices[i, 3]
        print(input_names[i].ljust(27), f"{sobol_lift: .4f} {sobol_torque: 6.4f}")
