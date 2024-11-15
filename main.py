import time
import argparse
from copy import deepcopy

from numpy import e
from numpy import pi

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Get mlx ADOPT optimizer from https://github.com/deboradum/mlx-ADOPT-optimizer
from ADOPT import ADOPT, ADOPTw

# Get mlx Amsgrad optimizer from https://github.com/deboradum/mlx-amsgrad-optimizer
from Amsgrad import Amsgrad

class OptimizerTest:
    def __init__(self):
        # self.true_optimal_loss =
        # self.true_optimal_point =
        # self.loss_margin =
        # self.point_margin =
        # self.name =
        raise NotImplementedError

    def loss_fn(self):
        return self.model()

    def test(self, optimizer, num_steps):
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        valid_loss = False
        valid_point = False
        start = time.perf_counter()
        for step in range(num_steps):
            loss, grads = loss_and_grad_fn()
            optimizer.update(self.model, grads)
            mx.eval(self.model, optimizer.state)
            if step % 1000 == 0:
                if (
                    mx.allclose(loss, self.true_optimal_loss, atol=self.loss_margin)
                    and not valid_loss
                ):
                    print(f"\tStep {step}: Optimal loss converged.")
                    valid_loss = True
                if (
                    mx.allclose(
                        self.model.X.weight,
                        self.true_optimal_point,
                        atol=self.point_margin,
                    )
                    and not valid_point
                ):
                    print(f"\tStep {step}: Optimal point converged.")
                    valid_point = True
                if valid_point and valid_loss:
                    break

        if not valid_point or not valid_loss:
            print(f"\tOptimizer did not converge within {num_steps} steps.")

            point = [str(x) for x in np.array(self.model.X.weight)]
            opt_point = [str(x) for x in np.array(self.true_optimal_point)]

            if not valid_point:
                print(
                    f"\tFinal point: ({', '.join(str(x) for x in point)}). Expected ({', '.join(opt_point)})",
                )
            if not valid_loss:
                print(f"\tFinal loss: {loss}. Expected {self.true_optimal_loss}")

        taken = round(time.perf_counter() - start, 2)
        print(f"\tTook {taken}s")

    def print_info(self):
        print(f"Testing {self.name} function...")


# https://www.sfu.ca/~ssurjano/rosen.html
class Rosenbrock(nn.Module):
    def __init__(self, initial=mx.array([-1.2, 1.0])):
        super().__init__()
        self.X = nn.Linear(1, 2, False)
        self.X.weight = initial

    def __call__(self):
        x1, x2 = self.X.weight
        return (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2


class RosenbrockTest(OptimizerTest):
    def __init__(self, initial=mx.array([-1.2, 1.0])):
        self.true_optimal_loss = mx.array(0)
        self.true_optimal_point = mx.ones_like(initial)
        self.loss_margin = 0.001
        self.point_margin = 0.01
        self.model = Rosenbrock()
        self.name = "Rosenbrock"


# https://www.sfu.ca/~ssurjano/rastr.html
class Rastrigin(nn.Module):
    def __init__(self, initial=mx.array([0.5, -0.3, 0.49, -0.4, -0.12])):
        super().__init__()
        self.n = initial.size
        self.X = nn.Linear(1, self.n, False)
        self.X.weight = initial

    def __call__(self):
        A = 10
        return A * self.n + mx.sum(
            mx.square(self.X.weight) - A * mx.cos(2 * pi * self.X.weight)
        )


class RastriginTest(OptimizerTest):
    def __init__(self, initial=mx.array([0.5, -0.3, 0.49, -0.4, -0.12])):
        self.true_optimal_loss = mx.array(0)
        self.true_optimal_point = mx.zeros_like(initial)
        self.loss_margin = 0.001
        self.point_margin = 0.001
        self.model = Rastrigin(initial)
        self.name = "Rastrigin"


# https://www.sfu.ca/~ssurjano/ackley.html
class Ackley(nn.Module):
    def __init__(self, initial=mx.array([0.62, -0.59])):
        super().__init__()
        self.X = nn.Linear(1, 2, False)
        self.X.weight = initial

    def __call__(self):
        x1, x2 = self.X.weight
        return (
            -20 * mx.exp(-0.2 * mx.sqrt(0.5 * (x1**2 + x2**2)))
            - mx.exp(0.5 * (mx.cos(2 * pi * x1) + mx.cos(2 * pi * x2)))
            + e
            + 20
        )


class AckleyTest(OptimizerTest):
    def __init__(self, initial=mx.array([0.62, -0.59])):
        self.true_optimal_loss = mx.array(0)
        self.true_optimal_point = mx.array([0, 0])
        self.loss_margin = 0.001
        self.point_margin = 0.01
        self.model = Ackley(initial)
        self.name = "Ackley"


def get_optimizer():
    supported_optimizers = {
        "adopt": ADOPT,
        "adoptw": ADOPTw,
        "adopt_b1": ADOPT,
        "amsgrad": Amsgrad,
        "amsgrad_b1": Amsgrad,
        "adam": optim.Adam,
        "adagrad": optim.Adagrad,
        "sgd": optim.SGD,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", type=str)
    parser.add_argument("lr", type=str)
    args = parser.parse_args()

    o = args.optimizer
    lr = float(args.lr)

    try:
        # I wanted to try out adopt with beta 1 decay
        if "_b1" in o:
            optimizer = supported_optimizers[o](lr, beta_decay=True)
        else:
            optimizer = supported_optimizers[o](lr)
        print(f"Testing {o} with a learning rate of {lr}\n")
        return optimizer
    except KeyError:
        print(
            f"Optimizer {o} is not supported, please pick one of the following:",
            ", ".join(supported_optimizers.keys()),
        )
        exit()


if __name__ == "__main__":
    mx.random.seed(42)
    num_steps = 20000

    optimizer = get_optimizer()

    for fun_test in [
        RosenbrockTest(),
        RastriginTest(),
        AckleyTest(),
    ]:
        o = deepcopy(optimizer)
        fun_test.print_info()
        fun_test.test(o, num_steps)
