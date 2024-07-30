from kaggle_environments import make, evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


import random


from connect_x.logger import get_logger


logger = get_logger(__name__)


class PolicyAgent(nn.Module):
    def __init__(self, input_dim=42, output_dim=7, embed_dim=256):
        super(PolicyAgent, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.head = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 2 * self.embed_dim),
            nn.BatchNorm1d(2 * self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2 * self.embed_dim, self.output_dim),
        )

    def forward(self, x):
        x = self.head(x)
        return F.softmax(x, dim=-1)


class ValueAgent(nn.Module):
    def __init__(self, input_dim=42, output_dim=1, embed_dim=256):
        super(ValueAgent, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.head = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 2 * self.embed_dim),
            nn.BatchNorm1d(2 * self.embed_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2 * self.embed_dim, self.output_dim),
        )

    def loss_fn(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def forward(self, x):
        x = self.head(x)
        return x


class Agent(nn.Module):

    def __init__(
        self,
        env=None,
        win_reward=2,
        draw_reward=1,
        loss_reward=-2,
        error_reward=-100,
        clip_ratio=0.2,
        discount=0.9,
        lmbda=0.9,
    ):
        super(Agent, self).__init__()
        self.env = env
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_reward = loss_reward
        self.error_reward = error_reward

        self.clip_ratio = clip_ratio

        self.policyagent = PolicyAgent()
        self.policyoptimizer = optim.AdamW(
            self.policyagent.parameters(), lr=0.001, betas=(0.5, 0.999)
        )
        self.valueagent = ValueAgent()
        self.valueoptimizer = optim.AdamW(
            self.valueagent.parameters(), lr=0.001, betas=(0.5, 0.999)
        )

        self.discount = discount
        self.lmbda = lmbda

    def train(
        self,
        game_counts=1001,
        every=100,
        switch=False,
        run_steps=4,
        epochs=5,
        epoch_number=0,
        verbose=False,
    ):

        logger.info("Training")
        total_rewards = []

        for game_count in tqdm(range(game_counts)):

            done = False
            states = []
            actions = []
            rewards = []
            old_policys = []
            one_game_rewards = []

            state = np.array(self.env.reset()["board"])
            while not done:
                self.policyagent.eval()
                action_probs = (
                    self.policyagent(torch.Tensor(state).reshape(1, 42))
                    .detach()
                    .numpy()[0]
                )
                action = np.random.choice(len(action_probs), p=action_probs)
                board_2d = state.reshape(6, 7)
                is_valid = any(board_2d[:, int(action)] == 0)

                if is_valid:
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.array(next_state["board"])
                else:
                    next_state = state
                    reward = None
                    done = True

                if done:
                    if reward is None:
                        reward = self.error_reward
                    elif reward == 1:
                        reward = self.win_reward
                    elif reward == -1:
                        reward = self.loss_reward
                    elif reward == 0:
                        reward = self.draw_reward
                else:
                    reward = -1 / 30

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_policys.append(action_probs)

                if (len(states) > run_steps) or (done and len(states) > 1):

                    states_tensor = torch.Tensor(states)

                    self.valueagent.eval()
                    values = self.valueagent(states_tensor).detach().numpy()
                    next_value = (
                        self.valueagent(torch.Tensor(next_state.reshape(1, -1)))
                        .detach()
                        .numpy()
                    )

                    rewards = np.array(rewards)
                    gaes = np.zeros_like(rewards)
                    n_steps_targets = np.zeros_like(rewards)

                    gae_sum = 0
                    forward_val = 0

                    if not done:
                        forward_val = next_value

                    for k in reversed(range(0, len(rewards))):
                        delta = rewards[k] + self.discount * forward_val - values[k]
                        gae_sum = self.discount * self.lmbda * gae_sum + delta
                        gaes[k] = gae_sum
                        forward_val = values[k]
                        n_steps_targets[k] = gaes[k] + values[k]

                    old_policys = self.policyagent(states_tensor).detach()

                    for epoch in range(epochs):

                        self.policyagent.train()
                        states_tensor = torch.Tensor(states)
                        self.policyoptimizer.zero_grad()
                        new_policys = self.policyagent(states_tensor)
                        action_one_hot = F.one_hot(
                            torch.Tensor(actions).long(), num_classes=7
                        ).detach()
                        old_p = torch.log(torch.sum(old_policys * action_one_hot))
                        new_p = torch.log(torch.sum(new_policys * action_one_hot))
                        ratio = torch.exp(old_p - new_p)
                        clip_ratio = torch.clip(
                            ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                        )
                        gaes_tensor = torch.Tensor(gaes).detach()
                        policy_agent_loss = -torch.mean(
                            torch.min(ratio * gaes_tensor, clip_ratio * gaes_tensor)
                        )
                        policy_agent_loss.backward()
                        self.policyoptimizer.step()
                        self.policyagent.eval()

                        self.valueagent.train()
                        states_tensor = torch.Tensor(states)
                        self.valueoptimizer.zero_grad()
                        value_agent_loss = self.valueagent.loss_fn(
                            torch.Tensor(n_steps_targets),
                            self.valueagent(states_tensor),
                        )
                        value_agent_loss.backward()
                        self.valueoptimizer.step()
                        self.valueagent.eval()
                    one_game_rewards += list(rewards)
                    states = []
                    actions = []
                    rewards = []
                    old_policys = []
                if done and len(states) == 1:
                    one_game_rewards += list(rewards)

                state = next_state

            total_rewards.append(one_game_rewards[-1])

            if (game_count % every == 0) and (game_count):
                if verbose:
                    print("-" * 30)
                    print(f"game_count:{game_count}")

                win_percent = np.mean(
                    np.array(total_rewards[game_count - every : game_count])
                    == self.win_reward
                )
                draw_percent = np.mean(
                    np.array(total_rewards[game_count - every : game_count])
                    == self.draw_reward
                )
                loss_percent = np.mean(
                    np.array(total_rewards[game_count - every : game_count])
                    == self.loss_reward
                )
                error_percent = np.mean(
                    np.array(total_rewards[game_count - every : game_count])
                    == self.error_reward
                )

                writer.add_scalar(
                    "Win/train", win_percent, game_count + (epoch_number * game_counts)
                )
                writer.add_scalar(
                    "Draw/train",
                    draw_percent,
                    game_count + (epoch_number * game_counts),
                )
                writer.add_scalar(
                    "Loss/train",
                    loss_percent,
                    game_count + (epoch_number * game_counts),
                )
                writer.add_scalar(
                    "Error/train",
                    error_percent,
                    game_count + (epoch_number * game_counts),
                )
                writer.add_scalar(
                    "PolicyAgentLoss/train",
                    policy_agent_loss,
                    game_count + (epoch_number * game_counts),
                )
                writer.add_scalar(
                    "ValueAgentLoss/train",
                    value_agent_loss,
                    game_count + (epoch_number * game_counts),
                )

                if verbose:
                    print(f"win_percent:{win_percent}")
                    print(f"draw_percent:{draw_percent}")
                    print(f"loss_percent:{loss_percent}")
                    print(f"error_percent:{error_percent}")
                    print(
                        f"policy_agent_loss:{policy_agent_loss},\nvalue_agent_loss:{value_agent_loss}"
                    )
                    print("-" * 30)

        writer.flush()
        writer.close()

    def predict(self, state):
        idx = torch.where(torch.sum(state.reshape(6, 7) == 0, axis=0) == 0)[0]
        self.policyagent.eval()
        with torch.no_grad():
            action_pros = self.policyagent(torch.Tensor(state)).detach().numpy()
            if len(idx):
                action_pros[0, idx] = 0
            action = np.argmax(action_pros)

        return int(action)
