import collections

import gym
import numpy as np
import torch

from lib.replay_buffer import ExperienceBuffer

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class Agent:
    def __init__(self, env: gym.Wrapper, test_env: gym.Wrapper, exp_buffer: ExperienceBuffer):
        self.env = env
        self.test_env = test_env
        self.exp_buffer = exp_buffer
        self._reset()
        self.episode = 0

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, nets: list, train_scores: list, epsilon=0.0, device="cpu"):
        """

        :param nets: list of K nets for adaptive-beta EBU
        :param epsilon:
        :param device:
        :return:
        """
        done_reward = None

        # TODO: not compatible with EpisodicLifeEnv!
        i = self.episode % len(nets)
        net = nets[i]

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([np.vstack(self.state)], copy=False).astype(np.float32) / 255.
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        train_scores[i] += reward

        # save exp in replay buffer
        exp = Experience(self.state.copy(), action, reward, is_done, new_state.copy())
        self.exp_buffer.append(exp)
        self.state = new_state

        # TODO: not compatible with episodic env_life!
        if is_done:
            self.episode += 1
            done_reward = self.total_reward
            self._reset()
        return done_reward

    @torch.no_grad()
    def play_test_episode(self, model, eps=0.05, episodes=30, max_steps=18_000, device='cpu'):
        scores = []
        for _ in range(episodes):
            is_done = False
            frame_idx = 0
            epi_reward = 0
            state = self.test_env.reset()

            while not is_done and frame_idx <= max_steps:
                if np.random.random() < eps:
                    action = self.test_env.action_space.sample()
                else:
                    state_a = np.array([np.vstack(state)], copy=False).astype(np.float32) / 255.
                    state_v = torch.tensor(state_a).to(device)
                    q_vals_v = model(state_v)
                    _, act_v = torch.max(q_vals_v, dim=1)
                    action = int(act_v.item())

                # do step in the test environment
                # TODO: not compatible with episodicLifeEnv
                new_state, reward, is_done, _ = self.test_env.step(action)
                epi_reward += reward

                state = new_state

            scores.append(epi_reward)

        return sum(scores) / len(scores)
