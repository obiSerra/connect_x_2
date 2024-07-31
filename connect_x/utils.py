import csv
import os.path
import random

import numpy as np
import torch
from kaggle_environments import evaluate


def get_outcomes(agent1, agent2, n_rounds=100):
    config = {"rows": 6, "columns": 7, "inarow": 4}
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds // 2)
    outcomes += [
        [b, a]
        for [a, b] in evaluate(
            "connectx", [agent2, agent1], config, [], n_rounds - n_rounds // 2
        )
    ]

    wins = outcomes.count([1, -1]) + outcomes.count([0, None])
    loses = outcomes.count([-1, 1]) + outcomes.count([None, 0])
    draws = n_rounds - wins - loses

    return wins, draws, loses


def print_outcomes(outcomes):
    print(f"Wins: {outcomes[0]}, Draws: {outcomes[1]}, Loses: {outcomes[2]}\n")


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_benchmark_row(filename, row, mode="a"):
    with open(filename, mode=mode) as file:
        writer = csv.writer(file)
        writer.writerow(row)


def benchmark_agent(agent_name, epoch, agent, n_rounds=100, write=True):
    benchmark_file = "models/benchmark.csv"
    fieldnames = [
        "agent_name",
        "epoch",
        "random-wins",
        "random-draws",
        "random-loses",
        "negamax-wins",
        "negamax-draws",
        "negamax-loses",
    ]

    if write and not os.path.exists(benchmark_file):
        write_benchmark_row(benchmark_file, fieldnames, mode="w")

    print(f"\n{agent_name} - Epoch {epoch} vs random")
    random_outcomes = get_outcomes(agent, "random", n_rounds)
    print_outcomes(random_outcomes)

    print(f"\n{agent_name} - Epoch {epoch} vs negamax")
    negamax_outcomes = get_outcomes(agent, "negamax", n_rounds)
    print_outcomes(negamax_outcomes)

    if write:
        write_benchmark_row(
            benchmark_file, (agent_name, epoch) + random_outcomes + negamax_outcomes
        )