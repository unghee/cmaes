# CMA-ES

[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](./LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/cmaes)](https://pypistats.org/packages/cmaes)

Lightweight Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [1] implementation.

![visualize-six-hump-camel](https://user-images.githubusercontent.com/5564044/73486622-db5cff00-43e8-11ea-98fb-8246dbacab6d.gif)

<details>
<summary>Rosenbrock function.</summary>

![visualize-rosenbrock](https://user-images.githubusercontent.com/5564044/73486620-dac46880-43e8-11ea-9295-ec0bfa774655.gif)

</details>

<details>
<summary>Warm starting CMA-ES [2].</summary>

| Rot Ellipsoid function | Quadratic function |
| ---------------------- | ------------------ |
| ![rot-ellipsoid](https://user-images.githubusercontent.com/5564044/106723051-0c01f500-664a-11eb-8f37-ece937a3e9a6.png) | ![quadratic](https://user-images.githubusercontent.com/5564044/106723041-09070480-664a-11eb-817a-b0292f2e0201.png) |

</details>

<details>
<summary>IPOP-CMA-ES [3] on Himmelblau function.</summary>

![visualize-ipop-cmaes-himmelblau](https://user-images.githubusercontent.com/5564044/88472274-f9e12480-cf4b-11ea-8aff-2a859eb51a15.gif)

</details>

<details>
<summary>BIPOP-CMA-ES [4] on Himmelblau function.</summary>

![visualize-bipop-cmaes-himmelblau](https://user-images.githubusercontent.com/5564044/88471815-55111800-cf48-11ea-8933-5a4b48c49eba.gif)

</details>

These GIF animations are generated by [cmaes_visualizer.py](tools/cmaes_visualizer.py) and [ws_cmaes_visualizer.py](tools/ws_cmaes_visualizer.py).


## News

* **2021/02/02** The paper "Warm Starting CMA-ES for Hyperparameter Optimization" written by [@nmasahiro](https://github.com/nmasahiro), the maintainer of this library, is accepted at AAAI 2021 :tada:
* **2020/07/29** Optuna's built-in CMA-ES sampler which uses this library under the hood is stabled at Optuna v2.0. Please check out the [v2.0 release blog](https://medium.com/optuna/optuna-v2-3165e3f1fc2).

## Installation

Supported Python versions are 3.6 or later.

```
$ pip install cmaes
```

Or you can install via [conda-forge](https://anaconda.org/conda-forge/cmaes).

```
$ conda install -c conda-forge cmaes
```

## Usage

This library provides an "ask-and-tell" style interface.

```python
import numpy as np
from cmaes import CMA

def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)
```

And you can use this library via [Optuna](https://github.com/optuna/optuna) [5], an automatic hyperparameter optimization framework.
Optuna's built-in CMA-ES sampler which uses this library under the hood is available from [v1.3.0](https://github.com/optuna/optuna/releases/tag/v1.3.0) and stabled at [v2.0.0](https://github.com/optuna/optuna/releases/tag/v2.2.0).
See [the documentation](https://optuna.readthedocs.io/en/stable/reference/samplers.html#optuna.samplers.CmaEsSampler) or [v2.0 release blog](https://medium.com/optuna/optuna-v2-3165e3f1fc2) for more details.

```python
import optuna

def objective(trial: optuna.Trial):
    x1 = trial.suggest_uniform("x1", -4, 4)
    x2 = trial.suggest_uniform("x2", -4, 4)
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

if __name__ == "__main__":
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=250)
```

<details>
<summary>Example of Warm-starting CMA-ES</summary>

Warm-starting CMA-ES (WS-CMA-ES) 

```python
import numpy as np
from cmaes import CMA, get_warm_start_mgd

def source_task(x1: float, x2: float) -> float:
    b = 0.4
    return (x1 - b) ** 2 + (x2 - b) ** 2

def target_task(x1: float, x2: float) -> float:
    b = 0.6
    return (x1 - b) ** 2 + (x2 - b) ** 2

if __name__ == "__main__":
    # Generate solutions from a source task
    source_solutions = []
    for _ in range(1000):
        x = np.random.random(2)
        value = source_task(x[0], x[1])
        source_solutions.append((x, value))

    # Estimate a promising distribution of the source task
    ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(
        source_solutions, gamma=0.1, alpha=0.1
    )
    optimizer = CMA(mean=ws_mean, sigma=ws_sigma, cov=ws_cov)

    # Run WS-CMA-ES
    print(" g    f(x1,x2)     x1      x2  ")
    print("===  ==========  ======  ======")
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = target_task(x[0], x[1])
            solutions.append((x, value))
            print(
                f"{optimizer.generation:3d}  {value:10.5f}"
                f"  {x[0]:6.2f}  {x[1]:6.2f}"
            )
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break
```

</details>

<details>
<summary>Example of IPOP-CMA-ES</summary>

You can easily implement IPOP-CMA-ES which restarts CMA-ES
with increasing population size.

```python
import math
import numpy as np
from cmaes import CMA

def ackley(x1, x2):
    # https://www.sfu.ca/~ssurjano/ackley.html
    return (
        -20 * math.exp(-0.2 * math.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))
        - math.exp(0.5 * (math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2)))
        + math.e + 20
    )

if __name__ == "__main__":
    bounds = np.array([[-32.768, 32.768], [-32.768, 32.768]])
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    mean = lower_bounds + (np.random.rand(2) * (upper_bounds - lower_bounds))
    sigma = 32.768 * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    for generation in range(200):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = ackley(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            # popsize multiplied by 2 (or 3) before each restart.
            popsize = optimizer.population_size * 2
            mean = lower_bounds + (np.random.rand(2) * (upper_bounds - lower_bounds))
            optimizer = CMA(mean=mean, sigma=sigma, population_size=popsize)
            print(f"Restart CMA-ES with popsize={popsize}")
```

</details>

<details>
<summary>Example of BIPOP-CMA-ES</summary>

Here is an example of BIPOP-CMA-ES which applies two interlaced restart strategies,
one with an increasing population size and one with varying small population sizes.

```python
import math
import numpy as np
from cmaes import CMA

def ackley(x1, x2):
    # https://www.sfu.ca/~ssurjano/ackley.html
    return (
        -20 * math.exp(-0.2 * math.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))
        - math.exp(0.5 * (math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2)))
        + math.e + 20
    )

if __name__ == "__main__":
    bounds = np.array([[-32.768, 32.768], [-32.768, 32.768]])
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    mean = lower_bounds + (np.random.rand(2) * (upper_bounds - lower_bounds))
    sigma = 32.768 * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    n_restarts = 0  # A small restart doesn't count in the n_restarts
    small_n_eval, large_n_eval = 0, 0
    popsize0 = optimizer.population_size
    inc_popsize = 2

    # Initial run is with "normal" population size; it is
    # the large population before first doubling, but its
    # budget accounting is the same as in case of small
    # population.
    poptype = "small"

    for generation in range(200):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = ackley(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            n_eval = optimizer.population_size * optimizer.generation
            if poptype == "small":
                small_n_eval += n_eval
            else:  # poptype == "large"
                large_n_eval += n_eval

            if small_n_eval < large_n_eval:
                poptype = "small"
                popsize_multiplier = inc_popsize ** n_restarts
                popsize = math.floor(
                    popsize0 * popsize_multiplier ** (np.random.uniform() ** 2)
                )
            else:
                poptype = "large"
                n_restarts += 1
                popsize = popsize0 * (inc_popsize ** n_restarts)

            mean = lower_bounds + (np.random.rand(2) * (upper_bounds - lower_bounds))
            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                bounds=bounds,
                population_size=popsize,
            )
            print("Restart CMA-ES with popsize={} ({})".format(popsize, poptype))
```

</details>


## Benchmark results

| [Rosenbrock function](https://www.sfu.ca/~ssurjano/rosen.html) | [Six-Hump Camel function](https://www.sfu.ca/~ssurjano/camel6.html) |
| ------------------- | ----------------------- |
| ![rosenbrock](https://user-images.githubusercontent.com/5564044/73486735-0cd5ca80-43e9-11ea-9e6e-35028edf4ee8.png) | ![six-hump-camel](https://user-images.githubusercontent.com/5564044/73486738-0e9f8e00-43e9-11ea-8e65-d60fd5853b8d.png) |

This implementation (green) stands comparison with [pycma](https://github.com/CMA-ES/pycma) (blue).
See [benchmark](./benchmark) for details.

## Links

**Other libraries:**

I respect all libraries involved in CMA-ES.

* [pycma](https://github.com/CMA-ES/pycma) : Most famous CMA-ES implementation by Nikolaus Hansen.
* [pymoo](https://github.com/msu-coinlab/pymoo) : Multi-objective optimization in Python.

**References:**

* [1] [N. Hansen, The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772, 2016.](https://arxiv.org/abs/1604.00772)
* [2] [Masahiro Nomura, Shuhei Watanabe, Youhei Akimoto, Yoshihiko Ozaki, Masaki Onishi. “Warm Starting CMA-ES for Hyperparameter Optimization”, AAAI. 2021.](https://arxiv.org/abs/2012.06932)
* [3] [Auger, A., Hansen, N.: A restart CMA evolution strategy with increasing population size. In: Proceedings of the 2005 IEEE Congress on Evolutionary Computation (CEC’2005), pp. 1769–1776 (2005a)](https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/EAMHCO/contributionsCEC05/auger05ARCMA.pdf)
* [4] [Hansen N. Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed. In the workshop Proceedings of the Genetic and Evolutionary Computation Conference, GECCO, pages 2389–2395. ACM, 2009.](https://hal.inria.fr/inria-00382093/document)
* [5] [Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama. 2019. Optuna: A Next-generation Hyperparameter Optimization Framework. In The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’19), August 4–8, 2019.](https://dl.acm.org/citation.cfm?id=3330701)
