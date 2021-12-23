import ray


ray.init("ray://localhost:10001")


import time

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.optuna import OptunaSearch


def evaluation_fn(step, width, height):
    return (0.1 + width * step / 100)**(-1) + height * 0.1


def easy_objective(config):
    # Hyperparameters
    width, height = config["width"], config["height"]

    for step in range(config["steps"]):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(step, width, height)
        # Feed the score back back to Tune.
        tune.report(iterations=step, mean_loss=intermediate_score)
        time.sleep(0.1)


def run_optuna_tune(smoke_test=False):
    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    scheduler = AsyncHyperBandScheduler()
    analysis = tune.run(
        easy_objective,
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=10 if smoke_test else 100,
        config={
            "steps": 100,
            "width": tune.uniform(0, 20),
            "height": tune.uniform(-100, 100),
            "activation": tune.choice(["relu", "tanh"])
        })

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    run_optuna_tune(smoke_test=True)