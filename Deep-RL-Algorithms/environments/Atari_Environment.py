from gymnasium import Wrapper, spaces
from .Open_AI_Wrappers import *


def make_minatar_game(env_id, max_episode_steps=None, render_mode=False):
    if render_mode:
        env = gym.make(env_id, render_mode='human')
    else:
        env = gym.make(env_id)
    env.frameskip = 1
    env = MinAtarNoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = wrap_minatar_deepmind(env)
    return env

def make_atari_game(env_id, max_episode_steps=None, render_mode=False):
    if render_mode:
        env = gym.make(env_id, render_mode='human')
    else:
        env = gym.make(env_id)
    env.frameskip = 1
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = wrap_deepmind(env)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True):
    """Configure environment for DeepMind-style Atari """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    env = SwapChannelOrder(env)
    return env

def wrap_minatar_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True):
    # if episode_life:
    #     env = EpisodicLifeEnv(env)
    if 'f' in env.game.env.action_map:
        env = MinAtarFireResetEnv(env)
    # env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    env = SwapChannelOrder(env)
    return env