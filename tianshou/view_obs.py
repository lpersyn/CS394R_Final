# Adpated from: https://github.com/aai-institute/tianshou/blob/master/examples/atari/
# Mostly out own work

import argparse
import datetime
import os
import pprint
import sys

import PIL.Image
import numpy as np
import torch
from atari_network import DQN
from atari_wrapper import make_atari_env

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import DiscreteSACPolicy, ICMPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
import warnings
import PIL 


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SpaceInvadersNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--noise-weight", type=float, default=0.0) # how much noise to add to the image
    return parser.parse_args()


def test_discrete_sac(args: argparse.Namespace = get_args()) -> None:
    args.watch = False
    env, train_envs, test_envs, watch_env = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
        noise=args.noise_weight,
        create_watch_env=(args.watch and args.render > 0),
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    images = []
    for _ in range(1):
        obs, _ = env.reset()
        frame = obs[0]
        image = PIL.Image.fromarray(frame)
        PIL.Image.Image.save(image, "./figures/states_with_diff_noise/test.png")
        # images.append(image)

    # Display the images
    # for image in images:
    #     image.show()
    


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    test_discrete_sac(get_args())