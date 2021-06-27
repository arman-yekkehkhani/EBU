#!/usr/bin/env python3

# __author__ = 'Maxim Lapan' # dqn
__author__ = 'arman-yekkehkhani'  # EBU Implementation

import argparse
import time
from distutils import util

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

DEFAULT_ENV_NAME = "BreakoutNoFrameskip-v4"
DEFAULT_TOTAL_STEPS = 20_000_000
METHOD = 'dqn'

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 1_000_000
LEARNING_RATE = 0.00025
SYNC_TARGET_FRAMES = 10_000
REPLAY_START_SIZE = 50_000
UPDATE_FREQ = 4
SYNC_K_NETS = 250_000
TEST_AGENT = SYNC_K_NETS

EPSILON_DECAY_LAST_FRAME = 1_000_000
EPSILON_START = 1.0
EPSILON_FINAL = 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--total_steps", type=int, default=DEFAULT_TOTAL_STEPS,
                        help="total steps of training")
    parser.add_argument("--method", type=str, default=METHOD,
                        help="Methods: dqn(default) or ebu")
    parser.add_argument("--beta", type=float, default=0.5, help="Diffusion factor")
    parser.add_argument("--log", type=util.strtobool, default=True, help="log training process in wandb")
    parser.add_argument("--k", type=int, default=1, help="number of betas")

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    env = wrappers.make_env(args.env)
    test_env = wrappers.make_env(args.env)
    env.seed(123)

    method = args.method
    print(args.env)
    print(method)

    if args.log:
        wandb.init(project=args.env, name=method, config={"gamma": GAMMA,
                                                          "beta": args.beta,
                                                          "k": args.k,
                                                          "batch size": BATCH_SIZE,
                                                          "replay size": REPLAY_SIZE,
                                                          "replay start size": REPLAY_START_SIZE,
                                                          "lr": LEARNING_RATE,
                                                          "sync target": SYNC_TARGET_FRAMES,
                                                          "min epsilon": EPSILON_FINAL,
                                                          "epsilon decay steps": EPSILON_DECAY_LAST_FRAME})

    if method == 'dqn' and args.k != 1:
        raise NotImplementedError

    K = args.k
    nets = [dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device) for i in range(K)]
    tgt_nets = [dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device) for j in range(K)]
    print(nets[0])

    train_scores = [0 for i in range(K)]

    if K != 1:
        betas = np.linspace(0, 1, K)
    else:
        betas = [args.beta]

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, test_env, buffer)
    epsilon = EPSILON_START

    optimizers = [optim.Adam(nets[i].parameters(), lr=LEARNING_RATE) for i in range(K)]
    ebu_trainer = EbuTrainer(optimizers, betas)

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while frame_idx < args.total_steps:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(nets, train_scores, epsilon, device=device)
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
            if args.log:
                wandb.log({'eps': epsilon,
                           'speed': speed,
                           'reward 100': mean_reward,
                           'reward': reward}, step=frame_idx)

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            for i in range(K):
                tgt_nets[i].load_state_dict(nets[i].state_dict())

        if frame_idx % UPDATE_FREQ == 0 and method == 'dqn':
            optimizers[0].zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, nets[0], tgt_nets[0], GAMMA, device=device)
            loss_t.backward()
            optimizers[0].step()

        if frame_idx % UPDATE_FREQ == 0 and method == 'ebu':
            ebu_trainer.ebu_train_step(nets, tgt_nets, env.action_space.n, buffer, BATCH_SIZE, device, gamma=GAMMA)

        if frame_idx % SYNC_K_NETS == 0 and method == 'ebu':
            best_i = np.argmax(train_scores)
            if args.log:
                wandb.log({'best_beta': best_i}, step=frame_idx)
            for i in range(K):
                nets[i].load_state_dict(nets[best_i].state_dict())
                tgt_nets[i].load_state_dict(tgt_nets[best_i].state_dict())
            train_scores = [0 for i in range(K)]

        if frame_idx % TEST_AGENT == 0:
            model = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
            model.load_state_dict(nets[best_i].state_dict())
            test_score = agent.play_test_episode(model, device=device)
            if args.log:
                wandb.log({'test_score': test_score}, step=frame_idx)
            del model
