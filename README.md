#  mlx-optimizer-testsuite
A simple test suite I wrote used for testing converging on my MLX implementations of [ADOPT](https://github.com/deboradum/mlx-ADOPT-optimizer) and [Amsgrad](https://github.com/deboradum/mlx-amsgrad-optimizer) optimizers.

To add more test, first create an implementation of the function (for example, the Rosenbrock function), derived from `mx.nn.Module`. Next, create a class derived from the `OptimizerTest` class, and initialise the desired true optimal point and true optimal loss, as well as the desired error margins. 

## Running tests
To run the testsuite, simply run `python main.py <optimizer> <learning_rate>`, for example: `python main.py "adam" "0.001"`
