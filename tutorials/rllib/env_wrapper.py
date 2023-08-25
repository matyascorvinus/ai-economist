# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Wrapper for making the gather-trade-build environment an OpenAI compatible environment.
This can then be used with reinforcement learning frameworks such as RLlib.
"""

import os
import pickle
import random
import warnings

import numpy as np
from ai_economist import foundation
from gym import spaces
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

_BIG_NUMBER = 1e20


def recursive_list_to_np_array(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, list):
                new_d[k] = np.array(v)
            elif isinstance(v, dict):
                new_d[k] = recursive_list_to_np_array(v)
            elif isinstance(v, (float, int, np.floating, np.integer)):
                new_d[k] = np.array([v])
            elif isinstance(v, np.ndarray):
                new_d[k] = v
            else:
                raise AssertionError
        return new_d
    raise AssertionError


def pretty_print(dictionary):
    for key in dictionary:
        print("{:15s}: {}".format(key, dictionary[key].shape))
    print("\n")


class RLlibEnvWrapper(MultiAgentEnv):
    """
    Environment wrapper for RLlib. It sub-classes MultiAgentEnv.
    This wrapper adds the action and observation space to the environment,
    and adapts the reset and step functions to run with RLlib.
    """

    def __init__(self, env_config, verbose=False):
        self.env_config_dict = env_config["env_config_dict"]

        # Adding env id in the case of multiple environments
        if hasattr(env_config, "worker_index"):
            self.env_id = (
                env_config["num_envs_per_worker"] * (env_config.worker_index - 1)
            ) + env_config.vector_index
        else:
            self.env_id = None

        self.env = foundation.make_env_instance(**self.env_config_dict)
        self.verbose = verbose
        self.sample_agent_idx = str(self.env.all_agents[0].idx)

        obs = self.env.reset()

        self.observation_space = self._dict_to_spaces_dict(obs["0"])
        self.observation_space_pl = self._dict_to_spaces_dict(obs["p"])

        if self.env.world.agents[0].multi_action_mode:
            self.action_space = spaces.MultiDiscrete(
                self.env.get_agent(self.sample_agent_idx).action_spaces
            )
            self.action_space.dtype = np.int64
            self.action_space.nvec = self.action_space.nvec.astype(np.int64)

        else:
            self.action_space = spaces.Discrete(
                self.env.get_agent(self.sample_agent_idx).action_spaces
            )
            self.action_space.dtype = np.int64

        if self.env.world.planner.multi_action_mode:
            self.action_space_pl = spaces.MultiDiscrete(
                self.env.get_agent("p").action_spaces
            )
            self.action_space_pl.dtype = np.int64
            self.action_space_pl.nvec = self.action_space_pl.nvec.astype(np.int64)

        else:
            self.action_space_pl = spaces.Discrete(
                self.env.get_agent("p").action_spaces
            )
            self.action_space_pl.dtype = np.int64

        self._seed = None
        if self.verbose:
            print("[EnvWrapper] Spaces")
            print("[EnvWrapper] Obs (a)   ")
            pretty_print(self.observation_space)
            print("[EnvWrapper] Obs (p)   ")
            pretty_print(self.observation_space_pl)
            print("[EnvWrapper] Action (a)", self.action_space)
            print("[EnvWrapper] Action (p)", self.action_space_pl)

    def _dict_to_spaces_dict(self, obs):
        dict_of_spaces = {}
        for k, v in obs.items():

            # list of lists are listified np arrays
            _v = v
            if isinstance(v, list):
                _v = np.array(v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                _v = np.array([v])

            # assign Space
            if isinstance(_v, np.ndarray):
                x = float(_BIG_NUMBER)
                # Warnings for extreme values
                if np.max(_v) > x:
                    warnings.warn("Input is too large!")
                if np.min(_v) < -x:
                    warnings.warn("Input is too small!")
                box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                # This loop avoids issues with overflow to make sure low/high are good.
                while not low_high_valid:
                    x = x // 2
                    box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                    low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                dict_of_spaces[k] = box

            elif isinstance(_v, dict):
                dict_of_spaces[k] = self._dict_to_spaces_dict(_v)
            else:
                raise TypeError
        return spaces.Dict(dict_of_spaces)

    @property
    def pickle_file(self):
        if self.env_id is None:
            return "game_object.pkl"
        return "game_object_{:03d}.pkl".format(self.env_id)

    def save_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "wb") as F:
            pickle.dump(self.env, F)

    def load_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "rb") as F:
            self.env = pickle.load(F)

    @property
    def n_agents(self):
        return self.env.n_agents

    @property
    def summary(self):
        last_completion_metrics = self.env.previous_episode_metrics
        if last_completion_metrics is None:
            return {}
        last_completion_metrics["completions"] = int(self.env._completions)
        return last_completion_metrics

    def get_seed(self):
        return int(self._seed)

    def seed(self, seed):
        # Using the seeding utility from OpenAI Gym
        # https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        _, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as an uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31

        if self.verbose:
            print(
                "[EnvWrapper] twisting seed {} -> {} -> {} (final)".format(
                    seed, seed1, seed2
                )
            )

        seed = int(seed2)
        np.random.seed(seed2)
        random.seed(seed2)
        self._seed = seed2

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return recursive_list_to_np_array(obs)

    def step(self, action_dict):
        obs, rew, done, info = self.env.step(action_dict)
        assert isinstance(obs[self.sample_agent_idx]["action_mask"], np.ndarray)

        return recursive_list_to_np_array(obs), rew, done, info

BIG_NUMBER = 1e20

import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiDiscrete

def recursive_obs_dict_to_spaces_dict(obs):
    """Recursively return the observation space dictionary
    for a dictionary of observations

    Args:
        obs (dict): A dictionary of observations keyed by agent index
        for a multi-agent environment

    Returns:
        Dict: A dictionary (space.Dict) of observation spaces
    """
    assert isinstance(obs, dict)
    dict_of_spaces = {}
    for k, v in obs.items():

        # list of lists are listified np arrays
        _v = v
        if isinstance(v, list):
            _v = np.array(v)
        elif isinstance(v, (int, np.integer, float, np.floating)):
            _v = np.array([v])

        # assign Space
        if isinstance(_v, np.ndarray):
            x = float(BIG_NUMBER)
            box = Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
            low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            # This loop avoids issues with overflow to make sure low/high are good.
            while not low_high_valid:
                x = x // 2
                box = Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            dict_of_spaces[k] = box

        elif isinstance(_v, dict):
            dict_of_spaces[k] = recursive_obs_dict_to_spaces_dict(_v)
        else:
            raise TypeError
    return Dict(dict_of_spaces)


class FoundationEnvWrapperRlib:
    """
    The environment wrapper class for Foundation.
    This wrapper determines whether the environment reset and steps happen on the
    CPU or the GPU, and proceeds accordingly.
    If the environment runs on the CPU, the reset() and step() calls also occur on
    the CPU.
    If the environment runs on the GPU, only the first reset() happens on the CPU,
    all the relevant data is copied over the GPU after, and the subsequent steps
    all happen on the GPU.
    """

    def __init__(
           self, env_config
    ):
        # Need to pass in an environment instance
        self.env_config_dict = env_config["env_config_dict"]

        self.env = foundation.make_env_instance(**self.env_config_dict)

        self.n_agents = self.env.num_agents
        self.episode_length = self.env.episode_length

        assert self.env.name
        self.name = self.env.name

        # Add observation space to the env
        # --------------------------------
        # Note: when the collated agent "a" is present, add obs keys
        # for each individual agent to the env
        # and remove the collated agent "a" from the observation
        obs = self.obs_at_reset()
        self.env.observation_space = recursive_obs_dict_to_spaces_dict(obs)
        self.observation_space = self.env.observation_space
        # Add action space to the env
        # ---------------------------
        self.env.action_space = {}
        for agent_id in range(len(self.env.world.agents)):
            if self.env.world.agents[agent_id].multi_action_mode:
                self.env.action_space[str([agent_id])] = MultiDiscrete(
                    self.env.get_agent(str(agent_id)).action_spaces
                )
            else:
                self.env.action_space[str(agent_id)] = Discrete(
                    self.env.get_agent(str(agent_id)).action_spaces
                )
            # self.env.action_space[str(agent_id)].dtype = np.int32

        if self.env.world.planner.multi_action_mode:
            self.env.action_space["p"] = MultiDiscrete(
                self.env.get_agent("p").action_spaces
            )
        else:
            self.env.action_space["p"] = Discrete(self.env.get_agent("p").action_spaces)
        # self.env.action_space["p"].dtype = np.int32

        # Ensure the observation and action spaces share the same keys
        assert set(self.env.observation_space.keys()) == set(
            self.env.action_space.keys()
        )
        self.action_space = self.env.get_agent(str(0)).action_spaces
        self.action_space_pl = self.env.action_space["p"]

        # CUDA-specific initializations
        # -----------------------------
        # Flag to determine whether to use CUDA or not

        # Flag to determine where the reset happens (host or device)
        # First reset is always on the host (CPU), and subsequent resets are on
        # the device (GPU)
        self.reset_on_host = True

        # Steps specific to GPU runs
        # --------------------------
    def reset_all_envs(self):
        """
        Reset the state of the environment to initialize a new episode.
        if self.reset_on_host is True:
            calls the CPU env to prepare and return the initial state
        if self.use_cuda is True:
            if self.reset_on_host is True:
                expands initial state to parallel example_envs and push to GPU once
                sets self.reset_on_host = False
            else:
                calls device hard reset managed by the CUDAResetter
        """
        self.env.world.timestep = 0

        if self.reset_on_host:
            # Produce observation
            obs = self.obs_at_reset()
        else:
            assert self.use_cuda

        if self.use_cuda:  # GPU version
            if self.reset_on_host:

                # Helper function to repeat data across the env dimension
                def repeat_across_env_dimension(array, num_envs):
                    return np.stack([array for _ in range(num_envs)], axis=0)

                # Copy host data and tensors to device
                # Note: this happens only once after the first reset on the host

                scenario_and_components = [self.env] + self.env.components

                for item in scenario_and_components:
                    # Add env dimension to data
                    # if "save_copy_and_apply_at_reset" is True
                    data_dictionary = item.get_data_dictionary()
                    tensor_dictionary = item.get_tensor_dictionary()
                    for key in data_dictionary:
                        if data_dictionary[key]["attributes"][
                            "save_copy_and_apply_at_reset"
                        ]:
                            data_dictionary[key]["data"] = repeat_across_env_dimension(
                                data_dictionary[key]["data"], self.n_envs
                            )

                    for key in tensor_dictionary:
                        if tensor_dictionary[key]["attributes"][
                            "save_copy_and_apply_at_reset"
                        ]:
                            tensor_dictionary[key][
                                "data"
                            ] = repeat_across_env_dimension(
                                tensor_dictionary[key]["data"], self.n_envs
                            )

                    self.cuda_data_manager.push_data_to_device(data_dictionary)

                    self.cuda_data_manager.push_data_to_device(
                        tensor_dictionary, torch_accessible=True
                    )

                # All subsequent resets happen on the GPU
                self.reset_on_host = False

                # Return the obs
                return obs
            # Returns an empty dictionary for all subsequent resets on the GPU
            # as arrays are modified in place
            self.env_resetter.reset_when_done(
                self.cuda_data_manager, mode="force_reset"
            )
            return {}
        return obs  # CPU version

    def reset_only_done_envs(self):
        """
        This function only works for GPU example_envs.
        It will check all the running example_envs,
        and only resets those example_envs that are observing done flag is True
        """
        assert self.use_cuda and not self.reset_on_host, (
            "reset_only_done_envs() only works "
            "for self.use_cuda = True and self.reset_on_host = False"
        )

        self.env_resetter.reset_when_done(self.cuda_data_manager, mode="if_done")
        return {}

    def step_all_envs(self, actions=None):
        """
        Step through all the environments' components and scenario
        """
        if self.use_cuda:
            # Step through each component
            for component in self.env.components:
                component.component_step()

            # Scenario step
            self.env.scenario_step()

            # Compute rewards
            self.env.generate_rewards()

            result = None  # Do not return anything
        else:
            assert actions is not None, "Please provide actions to step with."
            obs, rew, done, info = self.env.step(actions)
            obs = self._reformat_obs(obs)
            rew = self._reformat_rew(rew)
            result = obs, rew, done, info
        return result

    def obs_at_reset(self):
        """
        Calls the (Python) env to reset and return the initial state
        """
        obs = self.env.reset()
        obs = self._reformat_obs(obs)
        return obs

    def _reformat_obs(self, obs):
        if "a" in obs:
            # This means the env uses collated obs.
            # Set each individual agent as obs keys for processing with WarpDrive.
            for agent_id in range(self.env.n_agents):
                obs[str(agent_id)] = {}
                for key in obs["a"].keys():
                    obs[str(agent_id)][key] = obs["a"][key][..., agent_id]
            del obs["a"]  # remove the key "a"
        return obs

    def _reformat_rew(self, rew):
        if "a" in rew:
            # This means the env uses collated rew.
            # Set each individual agent as rew keys for processing with WarpDrive.
            assert isinstance(rew, dict)
            for agent_id in range(self.env.n_agents):
                rew[str(agent_id)] = rew["a"][agent_id]
            del rew["a"]  # remove the key "a"
        return rew

    def reset(self):
        """
        Alias for reset_all_envs() when CPU is used (conforms to gym-style)
        """
        return self.reset_all_envs()

    def step(self, actions=None):
        """
        Alias for step_all_envs() when CPU is used (conforms to gym-style)
        """
        return self.step_all_envs(actions)