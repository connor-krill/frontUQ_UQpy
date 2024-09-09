# UQpy at FrontUQ 2024

This repository is for the demonstration of [UQpy](https://uqpyproject.readthedocs.io/en/latest/) at the 
[FrontUQ 2024](https://www.frontuq-2024.com) conference.

## Setup

This repository is written entirely in Python.
We recommend you set up a new Python environment and
install the packages in `requirements.txt`, then run `validate_setup.py`
to confirm [UM-Bridge](https://um-bridge-benchmarks.readthedocs.io/en/docs/) 
and [UQpy](https://uqpyproject.readthedocs.io/en/latest/) are properly installed.
The script `validate_setup.py` runs the XFoil model twice, once with UM-Bridge and once with UQpy, and prints the results.

For simplicity, `validate_setup.py`

## XFoil Aerospace Model

The model is provided for us and computed using the package [xfoil](https://web.mit.edu/drela/Public/web/xfoil/).
The details of xfoil are not important to us, we treat this model as a black box.
The aerodynamic model takes in 5 inputs with distributions given by the table below.

| Parameter Name                 | Aleatory Distribution | Epistemic Interval |
|--------------------------------|-----------------------|--------------------|
| 1. Angle of Attack             | N(0, 0.1)             | [-0.3, 0.3]        |
| 2. Reynolds Number             | N(500,000, 2,500)     | [493,500, 507,500] |
| 3. Upper Surface Trip Location | N(0.3, 0.015)         | [0.225, 0.345]     |
| 4. Lower Surface Trip Location | N(0.7, 0.021)         | [0.637, 0.763]     |
| 5. Flap Deflection             | N(0, 0.08)            | [-0.24, 0.24]      |

The model returns the following 4 outputs:

1. Lift (CL)
2. Total Resistance (CD)
3. Resistance due to Pressure (CDp)
4. Torque (CM)


## Running the Model

The model is defined via a docker container provided by FrontUQ. 
The docker container can be found at `linusseelinger/xfoil_arm64`.
To run the docker container (`docker run`), detach it from the terminal session (`-d`), 
and specify port 4242 (`-p 4242`), run 

> docker run -d -p 4242 --name xfoil linusseelinger/xfoil_arm64 

Sometimes the port specification doesn't work, in which case you can view the port with either of the two following lines:

> docker port xfoil

> docker ps

With the docker container running, the following lines of python will run the model.
Make sure the port number after `localhost:` matches the port number returned by `docker port xfoil`.

```python
import umbridge
model = umbridge.HTTPModel("http://localhost:4242", "forward")
inputs = [[0.0, 500_000, 0.3, 0.7, 0.0]]
print(model(inputs))
```

### Running in Parallel

quick update: The images now support parallel model runs. Through

> docker run -e NUM_THREADS=10 -p 4242:4242 -it linusseelinger/xfoil

you can specify the maximum number of parallel model runs you'd like the server to do. 
Since each xfoil instance itself is still sequential, that should usually equal the number of CPU cores.
With a parallelized UQ software, the server will now actually handle parallel UM-Bridge requests in parallel.

# Authors

Shields Uncertainty Research Group    
Department of Civil and Systems Engineering  
Johns Hopkins University  
Baltimore, Maryland  