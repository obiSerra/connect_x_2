from kaggle_environments import make, evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from connect_x.agents import Agent
from connect_x.utils import get_outcomes, print_outcomes, seed_everything

config = {"rows": 6, "columns": 7, "inarow": 4}
env = make("connectx", config, debug=True)


seed_everything(seed=42)


def agent_max_reward(obs, config):
    import numpy as np
    import random

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


# games=[['random',None],[None,"negamax"],[agent_max_reward,None],[None,'random'],["negamax",None],[None,agent_max_reward]]
games = [["random", None]]

print(f"len(games):{len(games)}")
agent = Agent()
epochs = 1
for epoch in range(epochs):
    random_number = np.arange(len(games))
    np.random.shuffle(random_number)
    for i in range(len(games)):
        agent.env = env.train(games[random_number[i]])
        agent.train(game_counts=1001)


print_outcomes(get_outcomes(agent1=agent, agent2="random"))