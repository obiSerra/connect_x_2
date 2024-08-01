import argparse

import numpy as np
import torch
from kaggle_environments import make

from connect_x.agents import Agent
from connect_x.utils import benchmark_agent, seed_everything

config = {"rows": 6, "columns": 7, "inarow": 4}
env = make("connectx", config, debug=True)

seed_everything(seed=42)


def agent_max_reward(obs, config):
    import random

    import numpy as np

    def seed_everything(seed):
        random.seed(seed)

    seed_everything(seed=42)

    def get_reward(board, idx, rows=6, cols=7, label=1):
        total_reward = 0
        rewards = [1, 10, 30, 60, 100, 200]
        row_id, col_id = idx // rows, idx % cols
        if (row_id < rows - 1) and (board[(row_id + 1) * cols + col_id] == 0):
            return -1
        for position in [
            [row_id - 1, col_id - 1],
            [row_id - 1, col_id],
            [row_id - 1, col_id + 1],
            [row_id, col_id - 1],
            # [row_id,col_id+1],   [row_id+1,col_id-1],[row_id+1,col_id],[row_id+1,col_id+1]
        ]:
            cur_row, cur_col = row_id, col_id
            direction = [row_id - position[0], col_id - position[1]]
            point_count = 1
            while (
                (0 <= cur_row - direction[0] < rows)
                and (0 <= cur_col - direction[1] < cols)
                and (
                    board[(cur_row - direction[0]) * cols + (cur_col - direction[1])]
                    == label
                )
            ):
                point_count += 1

                cur_row = cur_row - direction[0]
                cur_col = cur_col - direction[1]
            cur_row, cur_col = row_id, col_id
            while (
                (0 <= cur_row + direction[0] < rows)
                and (0 <= cur_col + direction[1] < cols)
                and (
                    board[(cur_row + direction[0]) * cols + (cur_col + direction[1])]
                    == label
                )
            ):
                point_count += 1
                cur_row = cur_row + direction[0]
                cur_col = cur_col + direction[1]
            total_reward += rewards[point_count - 1]
        return total_reward

    board = obs.board
    total_rewards = np.zeros(42)
    for idx in range(len(board)):
        if board[idx] == 0:
            # label=1
            total_rewards[idx] = get_reward(board, idx=idx, label=1) + get_reward(
                board, idx=idx, label=2
            )
    max_reward_idx = np.argmax(total_rewards) % 7
    return int(max_reward_idx)


def agent_factory(agent):
    def agent_fn(obs, config):
        board = torch.Tensor(np.array(obs.board).reshape(1, -1))
        action = agent.predict(board)
        return action

    return agent_fn


def train_agent(
    agent_name, agent, env, games, epochs=1, game_counts=1000, verbose=False
):
    updated_games = games
    for epoch in range(epochs):
        random_number = np.arange(len(updated_games))
        np.random.shuffle(random_number)

        for i in range(len(updated_games)):
            print(f"Playing game {i + 1}/{len(updated_games)} - {str(updated_games[random_number[i]])}")
            agent.env = env.train(updated_games[random_number[i]])
            agent.train(
                game_counts=game_counts + 1,
                epoch_number=epoch,
                verbose=verbose,
                tot_games=len(updated_games) * game_counts,
                round_index=i,
            )

        agent.save(f"models/{agent_name}_{epoch}")
        print(f"Epoch: {epoch}\n")
        benchmark_agent(agent_name, epoch, agent_factory(agent), n_rounds=100)

        updated_games = games + [[agent_factory(agent), None], [None, agent_factory(agent)]]



def parse_args():
    parser = argparse.ArgumentParser(description="ConnectX")
    parser.add_argument(
        "task", type=str, help="Task to perform", choices=["train", "test"]
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default="agent_1",
        help="Name of the agent to be saved",
    )
    parser.add_argument(
        "--epoch", type=int, default=10, help="Number of epochs to train the agent"
    )
    parser.add_argument(
        "--game_counts",
        type=int,
        default=1000,
        help="Number of games to play in each epoch",
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="Print verbose logs"
    )
    return vars(parser.parse_args())


if __name__ == "__main__":

    opts = parse_args()

    if opts["task"] == "train":
        # games = [["random", None], [None, "random"], [None, "negamax"], ["negamax", None]]
        games = [["random", None], [None, "random"]]
        agent = Agent()
        train_agent(
            opts["agent_name"],
            agent,
            env,
            games,
            epochs=opts["epoch"],
            game_counts=opts["game_counts"],
            verbose=opts["verbose"],
        )
    elif opts["task"] == "test":
        agent_test = Agent()
        agent_test.load(f"models/{opts['agent_name']}_{opts['epoch']}")
        benchmark_agent(
            opts['agent_name'], opts['epoch'], agent_factory(agent_test), n_rounds=100, write=False
        )