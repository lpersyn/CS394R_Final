# We got it from: https://github.com/aai-institute/tianshou/blob/master/examples/atari/

import argparse
import datetime
import os
import pprint
import sys

import numpy as np
import torch
from atari_network import DQN
from atari_wrapper import make_atari_env

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from random_policy import RandomPolicy
from tianshou.policy import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
import warnings
from dataclasses import asdict


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SpaceInvadersNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=100000)
    # parser.add_argument("--actor-lr", type=float, default=1e-5)
    # parser.add_argument("--critic-lr", type=float, default=1e-5)
    # parser.add_argument("--gamma", type=float, default=0.99)
    # parser.add_argument("--n-step", type=int, default=3)
    # parser.add_argument("--tau", type=float, default=0.005)
    # parser.add_argument("--alpha", type=float, default=0.05)
    # parser.add_argument("--auto-alpha", action="store_true", default=False)
    # parser.add_argument("--alpha-lr", type=float, default=3e-4)
    # parser.add_argument("--epoch", type=int, default=100)
    # parser.add_argument("--step-per-epoch", type=int, default=100000)
    # parser.add_argument("--step-per-collect", type=int, default=10)
    # parser.add_argument("--update-per-step", type=float, default=0.1)
    # parser.add_argument("--batch-size", type=int, default=64)
    # parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    # parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--noise-weight", type=float, default=0.0) # how much noise to add to the image, 1.0 is good
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    # parser.add_argument(
    #     "--watch",
    #     default=False,
    #     action="store_true",
    #     help="watch the play of pre-trained policy only",
    # )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    return parser.parse_args()


def test_discrete_sac(args: argparse.Namespace = get_args()) -> None:
    args.watch = True
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
    # define model
    policy = RandomPolicy(action_space=env.action_space)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "random"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    # watch agent's performance, altered by Logan and Chloe
    def watch() -> None:
        env_to_run = test_envs
        if args.watch and args.render > 0:
            env_to_run = watch_env
        print("Setup watch envs ...")
        policy.eval()
        env_to_run.seed(args.seed) # test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(env_to_run), # buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
            )
            collector = Collector(policy, env_to_run, buffer, exploration_noise=True) # Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent, not saving buffer ...")
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(env_to_run), # buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
            )
            collector = Collector(policy, env_to_run, buffer, exploration_noise=True)
            result = collector.collect(n_episode=100, render=args.render, reset_before_collect=args.watch)
            # test_collector.reset()
            # result = test_collector.collect(n_episode=args.test_num, render=args.render)
        with open(os.path.join(log_path, "watch_result.txt"), "w") as f:
            pprint.pprint(asdict(result), stream=f)
        result.pprint_asdict()
    ####################################################

    watch()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    test_discrete_sac(get_args())