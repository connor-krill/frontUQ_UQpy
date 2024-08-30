import numpy as np
import umbridge
import UQpy as uq


def aerodynamic_model(x: np.ndarray) -> list:
    """Call the aerodynamic model with the help of UMBridge

    Note that UQpy assumes inputs are NumPy arrays, while UMBridge assumes inputs are lists.

    :param x: List containing a list of inputs. For example, ``[[1, 2, 3, 4, 5]]``
    :return:  List containing a list of outputs
    """
    umbridge_model = umbridge.HTTPModel(
        "  https://xfoil.linusseelinger.de:443", "forward"
    )
    return umbridge_model(x.tolist())


if __name__ == "__main__":
    inputs = [[0.0, 500_000, 0.3, 0.7, 0.0]]

    umbridge_model = umbridge.HTTPModel(
        "  https://xfoil.linusseelinger.de:443", "forward"
    )
    print("UM Bridge:", umbridge_model(inputs))

    uqpy_model = uq.RunModel(
        uq.PythonModel(
            model_script="quickstart.py", model_object_name="aerodynamic_model"
        )
    )
    uqpy_model.run(samples=inputs)
    print("UQpy:".ljust(10), uqpy_model.qoi_list[0])

