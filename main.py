#!/usr/bin/env python3

# __author__ = 'Maxim Lapan' # dqn
__author__ = 'arman-yekkehkhani'  # EBU Implementation

import argparse
import time

import numpy as np
import torch
import torch.optim as optim
import wandb

from dqn import calc_loss
from ebu import EbuTrainer
from lib import dqn_model
from lib import wrappers
from lib.agent import Agent
from lib.replay_buffer import ExperienceBuffer

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 15
BETA = 0.5
METHOD = 'dqn'

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 200_000
LEARNING_RATE = 0.0001
SYNC_TARGET_FRAMES = 1_000
REPLAY_START_SIZE = 10_000
UPDATE_FREQ = 1

EPSILON_DECAY_LAST_FRAME = 100_000
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    parser.add_argument("--method", type=str, default=METHOD,
                        help="Methods: dqn(default) or ebu")
    parser.add_argument("--beta", type=float, default=BETA, help="Diffusion factor")

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    # FIXME: use efficient_env instead of make_env
    env = wrappers.efficient_env(args.env)
    env.seed(123)

    method = args.method
    print(args.env)
    print(method)
    if args.beta:
        BETA = args.beta

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    wandb.init(project=args.env, name=method, config={"gamma": GAMMA,
                                                      "beta": BETA,
                                                      "batch size": BATCH_SIZE,
                                                      "replay size": REPLAY_SIZE,
                                                      "replay start size": REPLAY_START_SIZE,
                                                      "lr": LEARNING_RATE,
                                                      "sync target": SYNC_TARGET_FRAMES,
                                                      "min epsilon": EPSILON_FINAL,
                                                      "epsilon decay steps": EPSILON_DECAY_LAST_FRAME})
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    ebu_trainer = EbuTrainer(optimizer)

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            wandb.log({'eps': epsilon,
                       'speed': speed,
                       'reward 100': mean_reward,
                       'reward': reward}, step=frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        if frame_idx % UPDATE_FREQ == 0 and method == 'dqn':
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, GAMMA, device=device)
            loss_t.backward()
            optimizer.step()
        if frame_idx % UPDATE_FREQ == 0 and method == 'ebu':
            ebu_trainer.ebu_train_step(net, tgt_net, env.action_space.n, buffer, BATCH_SIZE,
                                       device, beta=BETA, gamma=GAMMA)
