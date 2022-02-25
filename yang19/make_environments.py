import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import MultiEnvs
import random
import numpy as np
import torch

def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if not cuda:
            raise ValueError("WARNING: You have a CUDA device, so you should probably run with --cuda")
    print(f"Set seed to {seed}")

def make_train_environment(args, seed):
    # Environment
    kwargs = {'dt': args.dt}
    # kwargs = {'dt': 100, 'sigma': 0, 'dim_ring': 2, 'cohs': [0.1, 0.3, 0.6, 1.0]}
    set_seed(seed, args.cuda)

    # Make supervised dataset
    tasks = ngym.get_collection('yang19')
    envs = [gym.make(task, **kwargs) for task in tasks]
    schedule = RandomSchedule(len(envs)) #TODO check for seed propagation
    env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
    dataset = ngym.Dataset(env, batch_size=args.batch_size, seq_len=args.seq_len)

    env = dataset.env
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    return env, dataset, ob_size, act_size

def make_analyze_environment(args, seed):
    set_seed(seed, args.cuda)
    # Environment
    timing = {'fixation': ('constant', 500)}
    kwargs = {'dt': args.dt, 'timing': timing}
    tasks = ngym.get_collection('yang19')
    envs = [gym.make(task, **kwargs) for task in tasks]
    env = MultiEnvs(envs, env_input=True)
    return env

