from typing import Tuple, List

import numpy as np
from cmaes._sepcma2 import SepCMA


def ellipsoid(x):
    n = len(x)
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    return sum([(1000 ** (i / (n - 1)) * x[i]) ** 2 for i in range(n)])


def main():
    dim = 40
    population_size = 10
    optimizers = [
        SepCMA(mean=np.array([3], dtype=float), n_dim=dim, sigma=2.0, population_size=population_size)
        for _ in range(dim)
    ]
    print(" evals    f(x)")
    print("======  ==========")

    evals = 0
    for _ in range(1000):
        solutions: List[Tuple[np.ndarray, float]] = []
        for _ in range(10):
            x = np.array([
                optimizer.ask()[0]
                for optimizer in optimizers
            ], dtype=float)

            value = ellipsoid(x)
            evals += 1
            solutions.append((x, value))
            if evals % 3000 == 0:
                print(f"{evals:5d}  {value:10.5f}")

        for i, optimizer in enumerate(optimizers):
            tell_solutions = [(np.array([s[0][i]]), s[1]) for s in solutions]
            optimizer.tell(tell_solutions)


if __name__ == "__main__":
    main()
