import numpy as np
import umbridge


def xfoil(x: np.ndarray) -> list:
    """Call the aerodynamic model with the help of UMBridge

    Note that UQpy assumes inputs are NumPy arrays, while UMBridge assumes inputs are lists.

    :param x: List containing a list of inputs. For example, ``[[1, 2, 3, 4, 5]]``
    :return:  List containing a list of outputs
    """
    umbridge_model = umbridge.HTTPModel("http://localhost:53185", "forward")
    output = umbridge_model(x.tolist())
    return np.array(output).squeeze()
