#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 07:26:31 2020

@author: manuel

/home/.../shaping_run.py --folder test  --seed 4 --alg A2C --stages 3 4

"""
import os
from ops.utils import get_name_and_command_from_dict as gncfd
from ops.utils import rest_arg_parser
import sys
import numpy as np
import glob
import importlib
import argparse
from copy import deepcopy as deepc
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Input

sys.path.append(os.path.expanduser("~/gym"))
sys.path.append(os.path.expanduser("~/stable-baselines"))
sys.path.append(os.path.expanduser("~/neurogym"))
sys.path.append(os.path.expanduser("~/multiple_choice"))
import get_activity as ga
import gym
import neurogym as ngym  # need to import it so ngym envs are registered
from neurogym.utils import plotting
from neurogym.wrappers import ALL_WRAPPERS
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import CheckpointCallback


def define_model(seq_len, num_h, obs_size, act_size, batch_size,
                 stateful, loss):
    """
    https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html
    """
    inp = Input(batch_shape=(batch_size, seq_len, obs_size), name="input")

    rnn = LSTM(num_h, return_sequences=True, stateful=stateful,
               name="RNN")(inp)

    dens = TimeDistributed(Dense(act_size, activation='softmax',
                                 name="dense"))(rnn)
    model = Model(inputs=[inp], outputs=[dens])

    model.compile(loss=loss, optimizer="Adam",  metrics=['accuracy'])

    return model


def test_env(env, kwargs, num_steps=100):
    """Test if all one environment can at least be run."""
    env = gym.make(env, **kwargs)
    env.reset()
    for stp in range(num_steps):
        action = 0
        state, rew, done, info = env.step(action)
        if done:
            env.reset()
    return env


def apply_wrapper(env, wrap_string, params):
    wrap_str = ALL_WRAPPERS[wrap_string]
    wrap_module = importlib.import_module(wrap_str.split(":")[0])
    wrap_method = getattr(wrap_module, wrap_str.split(":")[1])
    return wrap_method(env, **params)


def arg_parser():
    """
    Create an argparse.ArgumentParser for neuro environments
    """
    aadhf = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=aadhf)
    parser.add_argument('--dt', help='time step for environment', type=int,
                        default=None)
    parser.add_argument('--folder', help='where to save the data and model',
                        type=str, default=None)
    parser.add_argument('--train_mode', help='RL or SL',
                        type=str, default=None)
    parser.add_argument('--alg', help='RL algorithm to use',
                        type=str, default=None)
    parser.add_argument('--task', help='task', type=str, default=None)
    parser.add_argument('--num_trials',
                        help='number of trials to train', type=int,
                        default=None)
    parser.add_argument('--n_lstm',
                        help='number of units in the network',
                        type=int, default=None)
    parser.add_argument('--rollout',
                        help='rollout used to train the network',
                        type=int, default=None)
    parser.add_argument('--seed', help='seed for task',
                        type=int, default=None)
    parser.add_argument('--num_thrds', help='number of threads',
                        type=int, default=None)
    parser.add_argument('--n_ch', help='number of choices',
                        type=int, default=None)

    # n-alternative task params
    parser.add_argument('--stim_scale', help='stimulus evidence',
                        type=float, default=None)
    parser.add_argument('--ob_nch', help='Whether to provide num. channels',
                        type=bool, default=None)
    parser.add_argument('--zero_irrelevant_stim', help='Whether to zero' +
                        ' irrelevant stimuli', type=bool, default=None)

    # CV-learning task
    parser.add_argument('--th_stage',
                        help='threshold to change the stage',
                        type=float, default=None)
    parser.add_argument('--keep_days',
                        help='minimum number of sessions to spend on a stage',
                        type=int, default=None)
    parser.add_argument('--stages',
                        help='stages used for training',
                        type=int, nargs='+', default=None)

    # trial_hist wrapper parameters
    parser.add_argument('--probs', help='prob of main transition in the ' +
                        'n-alt task with trial hist.', type=float,
                        default=None)
    parser.add_argument('--block_dur',
                        help='dur. of block in the trial-hist wrappr (trials)',
                        type=int, default=None)
    parser.add_argument('--num_blocks', help='number of blocks', type=int,
                        default=None)
    parser.add_argument('--rand_blcks', help='whether transition matrix is' +
                        ' built randomly', type=bool, default=None)
    parser.add_argument('--blk_ch_prob', help='prob of trans. mat. change',
                        type=float, default=None)
    parser.add_argument('--balanced_probs', help='whether transition matrix is' +
                        ' side-balanced', type=bool, default=None)

    # trial_hist evolution wrapper parameters
    parser.add_argument('--ctx_dur',
                        help='dur. of context in the trial-hist wrappr (trials)',
                        type=int, default=None)
    parser.add_argument('--num_contexts', help='number of contexts', type=int,
                        default=None)
    parser.add_argument('--ctx_ch_prob', help='prob of trans. mat. change',
                        type=float, default=None)
    parser.add_argument('--death_prob', help='prob. of starting next generation',
                        type=float, default=None)
    parser.add_argument('--fix_2AFC', help='whether 2AFC is included in tr. mats',
                        type=bool, default=None)

    # variable-nch wrapper parameters
    parser.add_argument('--block_nch',
                        help='dur. of blck in the variable-nch wrapper (trials)',
                        type=int, default=None)
    parser.add_argument('--blocks_probs', help='probability of each block',
                        type=float, nargs='+', default=None)

    # monitor wrapper parameters
    parser.add_argument('--sv_fig',
                        help='indicates whether to save step-by-step figures',
                        type=bool, default=None)
    parser.add_argument('--sv_per',
                        help='number of trials to save behavioral data',
                        type=int, default=None)
    return parser


def update_dict(dict1, dict2):
    dict1.update((k, dict2[k]) for k in set(dict2).intersection(dict1))


def make_env(env_id, rank, seed=0, wrapps={}, **kwargs):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        for wrap in wrapps.keys():
            if not (wrap == 'Monitor-v0' and rank != 0):
                env = apply_wrapper(env, wrap, wrapps[wrap])
        return env
    set_global_seeds(seed)
    return _init


def run(alg, alg_kwargs, task, task_kwargs, wrappers_kwargs, expl_params,
        rollout, num_trials, folder, n_thrds, n_lstm, rerun=False,
        test_kwargs={}, num_retrains=10, seed=0, train_mode=None, sl_kwargs=None):
    train_mode = train_mode or 'RL'
    env = test_env(task, kwargs=task_kwargs, num_steps=1000)
    num_timesteps = int(1000 * num_trials / (env.num_tr))
    files = glob.glob(folder+'/*model*')
    vars_ = {'alg': alg, 'alg_kwargs': alg_kwargs, 'task': task,
             'task_kwargs': task_kwargs, 'wrappers_kwargs': wrappers_kwargs,
             'expl_params': expl_params, 'rollout': rollout, 'folder': folder,
             'num_trials': num_trials, 'n_thrds': n_thrds, 'n_lstm': n_lstm}
    np.savez(folder + '/params.npz', **vars_)
    if len(files) == 0 or rerun:
        if train_mode == 'RL':
            if alg == "A2C":
                from stable_baselines import A2C as algo
            elif alg == "ACER":
                from stable_baselines import ACER as algo
            elif alg == "ACKTR":
                from stable_baselines import ACKTR as algo
            elif alg == "PPO2":
                from stable_baselines import PPO2 as algo
            env = SubprocVecEnv([make_env(env_id=task, rank=i, seed=seed,
                                          wrapps=wrappers_kwargs, **task_kwargs)
                                 for i in range(n_thrds)])
            model = algo(LstmPolicy, env, verbose=0, n_steps=rollout,
                         n_cpu_tf_sess=n_thrds, tensorboard_log=None,
                         policy_kwargs={"feature_extraction": "mlp",
                                        "n_lstm": n_lstm},
                         **alg_kwargs)
            # this assumes 1 trial ~ 10 steps
            sv_freq = 5*wrappers_kwargs['Monitor-v0']['sv_per']
            chckpnt_cllbck = CheckpointCallback(save_freq=sv_freq,
                                                save_path=folder,
                                                name_prefix='model')
            model.learn(total_timesteps=num_timesteps, callback=chckpnt_cllbck)
            model.save(f"{folder}/model_{num_timesteps}_steps.zip")
            plotting.plot_rew_across_training(folder=folder)
        elif train_mode == 'SL':
            stps_ep = sl_kwargs['steps_per_epoch']
            wraps_sl = deepc(wrappers_kwargs)
            del wraps_sl['PassAction-v0']
            del wraps_sl['PassReward-v0']
            del wraps_sl['Monitor-v0']
            env = make_env(env_id=task, rank=0, seed=seed, wrapps=wraps_sl,
                           **task_kwargs)()
            dataset = ngym.Dataset(env, batch_size=sl_kwargs['btch_s'],
                                   seq_len=rollout, batch_first=True)
            obs_size = env.observation_space.shape[0]
            act_size = env.action_space.n
            model = define_model(seq_len=rollout, num_h=n_lstm, obs_size=obs_size,
                                 act_size=act_size, batch_size=sl_kwargs['btch_s'],
                                 stateful=sl_kwargs['stateful'],
                                 loss=sl_kwargs['loss'])
            # Train network
            data_generator = (dataset()
                              for i in range(stps_ep))
            model.fit(data_generator, verbose=1,
                      steps_per_epoch=stps_ep)
            model.save(f"{folder}/model_{stps_ep}_steps")

    if len(test_kwargs) != 0:
        for key in test_kwargs.keys():
            sv_folder = folder + key
            test_kwargs[key]['seed'] = seed
            if train_mode == 'RL':
                ga.get_activity(folder, alg, sv_folder, **test_kwargs[key])
            elif train_mode == 'SL':
                stps_ep = sl_kwargs['steps_per_epoch']
                wraps_sl = deepc(wrappers_kwargs)
                wraps_sl.update(test_kwargs[key]['wrappers'])
                del wraps_sl['PassAction-v0']
                del wraps_sl['PassReward-v0']
                env = make_env(env_id=task, rank=0, seed=seed, wrapps=wraps_sl,
                               **task_kwargs)()
                obs_size = env.observation_space.shape[0]
                act_size = env.action_space.n
                model_test = define_model(seq_len=1, batch_size=1,
                                          obs_size=obs_size, act_size=act_size,
                                          stateful=sl_kwargs['stateful'],
                                          num_h=n_lstm,
                                          loss=sl_kwargs['loss'])
                ld_f = folder+'model_'+str(stps_ep)+'_steps'.replace('//', '/')
                model_test.load_weights(ld_f)
                env.reset()
                for ind_stp in range(sl_kwargs['test_steps']):
                    obs = env.ob_now
                    obs = obs[np.newaxis]
                    obs = obs[np.newaxis]
                    action = model_test.predict(obs)
                    action = np.argmax(action, axis=-1)[0]
                    _, _, _, _ = env.step(action)


if __name__ == "__main__":
    # get params from call
    n_arg_parser = arg_parser()
    expl_params, unknown_args = n_arg_parser.parse_known_args(sys.argv)
    unkown_params = rest_arg_parser(unknown_args)
    if unkown_params:
        print('Unkown parameters: ', unkown_params)
    expl_params = vars(expl_params)
    expl_params = {k: expl_params[k] for k in expl_params.keys()
                   if expl_params[k] is not None}
    main_folder = expl_params['folder'] + '/'
    name, _ = gncfd(expl_params)
    instance_folder = main_folder + name + '/'
    # this is done wo the monitor wrapper's parameter folder is updated
    expl_params['folder'] = instance_folder
    if not os.path.exists(instance_folder):
        os.makedirs(instance_folder)
    # load parameters
    print('Main folder: ', main_folder)
    sys.path.append(os.path.expanduser(main_folder))
    spec = importlib.util.spec_from_file_location("params",
                                                  main_folder+"/params.py")
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    # update general params
    gen_params = params.general_params
    update_dict(gen_params, expl_params)
    # update task params
    task_params = params.task_kwargs[gen_params['task']]
    update_dict(task_params, expl_params)
    # update wrappers params
    wrappers_kwargs = params.wrapps
    for wrap in wrappers_kwargs.keys():
        if not wrap == 'Monitor-v0':
            params_temp = wrappers_kwargs[wrap]
            update_dict(params_temp, expl_params)
    # get params
    train_mode = gen_params['train_mode']
    task = gen_params['task']
    alg = gen_params['alg']
    alg_kwargs = params.algs[alg]
    seed = int(gen_params['seed'])
    num_trials = int(gen_params['num_trials'])
    rollout = int(gen_params['rollout'])
    num_thrds = int(gen_params['num_thrds'])
    n_lstm = int(gen_params['n_lstm'])
    task_kwargs = params.task_kwargs[gen_params['task']]
    if hasattr(params, 'test_kwargs'):
        test_kwargs = params.test_kwargs
    else:
        test_kwargs = {}
    run(alg=alg, alg_kwargs=alg_kwargs, task=task, task_kwargs=task_kwargs,
        wrappers_kwargs=params.wrapps, expl_params=expl_params, rollout=rollout,
        num_trials=num_trials, folder=instance_folder, n_thrds=num_thrds,
        n_lstm=n_lstm, test_kwargs=test_kwargs, seed=seed, train_mode=train_mode,
        sl_kwargs=params.sl_kwargs)
