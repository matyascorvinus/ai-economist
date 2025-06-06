# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime

import GPUtil
import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

try:
    num_gpus_available = len(GPUtil.getAvailable())
    print(f"Inside covid19_components.py: {num_gpus_available} GPUs are available.")
    if num_gpus_available == 0:
        print("No GPUs found! Running the simulation on a CPU.")
    else:
        from warp_drive.utils.constants import Constants
        from warp_drive.utils.data_feed import DataFeed

        _OBSERVATIONS = Constants.OBSERVATIONS
        _ACTIONS = Constants.ACTIONS
except ModuleNotFoundError:
    print(
        "Warning: The 'WarpDrive' package is not found and cannot be used! "
        "If you wish to use WarpDrive, please run "
        "'pip install rl-warp-drive' first."
    )
except ValueError:
    print("No GPUs found! Running the simulation on a CPU.")


@component_registry.add
class ControlUSStateOpenCloseStatus(BaseComponent):
    """
    Sets the open/close stringency levels for states.
    Args:
        n_stringency_levels (int): number of stringency levels the states can chose
            from. (Must match the number in the model constants dictionary referenced by
            the parent scenario.)
        action_cooldown_period (int): action cooldown period in days.
            Once a stringency level is set, the state(s) cannot switch to another level
            for a certain number of days (referred to as the "action_cooldown_period")
    """

    name = "ControlUSStateOpenCloseStatus"
    required_entities = []
    agent_subclasses = ["BasicMobileAgent"]

    # Covid-19 lockdown has effected the economy, so we need to add a reduction multiplier to the GDP
    # Assume at max level of stringency - level 10, the GDP is reduced by 30%, so each level increases 
    # will increase the GDP reduction multiplier by 3%, or 0.03

    def __init__(
        self,
        *base_component_args,
        n_stringency_levels=10,
        action_cooldown_period=28,
        reduced_gdp_multiplier_per_year=0.01,
        **base_component_kwargs,
    ):

        self.action_cooldown_period = action_cooldown_period
        super().__init__(*base_component_args, **base_component_kwargs)
        self.np_int_dtype = np.int32

        self.n_stringency_levels = int(n_stringency_levels)
        self.reduced_gdp_multiplier_per_year = float(reduced_gdp_multiplier_per_year)
        assert self.n_stringency_levels >= 2
        self._checked_n_stringency_levels = False

        self.masks = dict()
        self.default_agent_action_mask = [1 for _ in range(self.n_stringency_levels)]
        self.no_op_agent_action_mask = [0 for _ in range(self.n_stringency_levels)]
        # self.masks["a"] = np.repeat(
        #     np.array(self.no_op_agent_action_mask)[:, np.newaxis],
        #     self.n_agents,
        #     axis=-1,
        # )
        for agent in self.world.agents:
            
            self.masks[agent.idx] = self.no_op_agent_action_mask

        # (This will be overwritten during reset; see below)
        self.action_in_cooldown_until = None

    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        # Store the times when the next set of actions can be taken.
        self.action_in_cooldown_until = np.array(
            [self.world.timestep for _ in range(self.n_agents)]
        )

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return self.n_stringency_levels
        return None

    def generate_masks(self, completions=0):
        # for agent in self.world.agents:
        #     if self.world.use_real_world_policies:
        #         self.masks["a"][:, agent.idx] = self.default_agent_action_mask
        #     else:
        #         if self.world.timestep < self.action_in_cooldown_until[agent.idx]:
        #             # Keep masking the actions
        #             self.masks["a"][:, agent.idx] = self.no_op_agent_action_mask
        #         else:  # self.world.timestep == self.action_in_cooldown_until[agent.idx]
        #             # Cooldown period has ended; unmask the "subsequent" action
        #             self.masks["a"][:, agent.idx] = self.default_agent_action_mask
        # return self.masks
        for agent in self.world.agents:
            
            if self.world.use_real_world_policies or self.world.state_governments_policies_only:
                self.masks[agent.idx] = self.default_agent_action_mask
            else:
                if self.world.timestep < self.action_in_cooldown_until[agent.idx]:
                    # Keep masking the actions
                    self.masks[agent.idx] = self.no_op_agent_action_mask
                else:  # self.world.timestep == self.action_in_cooldown_until[agent.idx]
                    # Cooldown period has ended; unmask the "subsequent" action
                    self.masks[agent.idx] = self.default_agent_action_mask
        return self.masks


    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the GPU (device).
        """
        data_dict = DataFeed()
        data_dict.add_data(
            name="action_cooldown_period",
            data=self.action_cooldown_period,
        )
        # reduced_gdp_multiplier_per_year
        data_dict.add_data(
            name="reduced_gdp_multiplier_per_year",
            data=self.reduced_gdp_multiplier_per_year,
        )

        data_dict.add_data(
            name="action_in_cooldown_until",
            data=self.action_in_cooldown_until,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="num_stringency_levels",
            data=self.n_stringency_levels,
        )
        data_dict.add_data(
            name="default_agent_action_mask",
            data=[1] + self.default_agent_action_mask,
        )
        data_dict.add_data(
            name="no_op_agent_action_mask",
            data=[1] + self.no_op_agent_action_mask,
        )
        return data_dict

    def get_tensor_dictionary(self):
        """
        Create a dictionary of (Pytorch-accessible) data to push to the GPU (device).
        """
        tensor_dict = DataFeed()
        return tensor_dict

    def component_step(self):
        if self.world.use_cuda:
            self.world.cuda_component_step[self.name](
                self.world.cuda_data_manager.device_data("stringency_level"),
                self.world.cuda_data_manager.device_data("action_cooldown_period"),
                self.world.cuda_data_manager.device_data("action_in_cooldown_until"),
                self.world.cuda_data_manager.device_data("default_agent_action_mask"),
                self.world.cuda_data_manager.device_data("no_op_agent_action_mask"),
                self.world.cuda_data_manager.device_data("num_stringency_levels"),
                self.world.cuda_data_manager.device_data(f"{_ACTIONS}_a"),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_{self.name}-agent_policy_indicators"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_action_mask"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_{self.name}-agent_policy_indicators"
                ),
                self.world.cuda_data_manager.device_data("_timestep_"),
                self.world.cuda_data_manager.meta_info("n_agents"),
                self.world.cuda_data_manager.meta_info("episode_length"),
                block=self.world.cuda_function_manager.block,
                grid=self.world.cuda_function_manager.grid,
            )
        else:
            if not self._checked_n_stringency_levels:
                if self.n_stringency_levels != self.world.n_stringency_levels:
                    raise ValueError(
                        "The environment was not configured correctly. For the given "
                        "model fit, you need to set the number of stringency levels to "
                        "be {}".format(self.world.n_stringency_levels)
                    )
                self._checked_n_stringency_levels = True

            for agent in self.world.agents:
                # if(agent.idx == 50): 
                #     continue
                if self.world.use_real_world_policies or self.world.state_governments_policies_only:
                    # Use the action taken in the previous timestep
                    action = self.world.real_world_stringency_policy[
                        self.world.timestep - 1, agent.idx
                    ]
                else:
                    action = agent.get_component_action(self.name)
                assert 0 <= action <= self.n_stringency_levels

                # We only update the stringency level if the action is not a NO-OP.
                self.world.global_state["Stringency Level"][
                    self.world.timestep, agent.idx
                ] = (
                    self.world.global_state["Stringency Level"][
                        self.world.timestep - 1, agent.idx
                    ]
                    * (action == 0)
                    + action
                )

                # Average stringency level is a number calculated from Stringency Level

                # Update the agent's state
                agent.state[
                    "Current Open Close Stringency Level"
                ] = self.world.global_state["Stringency Level"][
                    self.world.timestep, agent.idx
                ]

                # Check if the action cooldown period has ended, and set the next
                # time until action cooldown. If current action is a no-op
                # (i.e., no new action was taken), the agent can take an action
                # in the very next step, otherwise it needs to wait for
                # self.action_cooldown_period steps. When in the action cooldown
                # period, whatever actions the agents take are masked out,
                # so it's always a NO-OP (see generate_masks() above)
                # The logic below influences the action masks.
                if self.world.timestep == self.action_in_cooldown_until[agent.idx] + 1:
                    if action == 0:  # NO-OP
                        self.action_in_cooldown_until[agent.idx] += 1
                    else:
                        self.action_in_cooldown_until[
                            agent.idx
                        ] += self.action_cooldown_period
            
            self.world.global_state["Average Stringency Level"] = np.mean(self.world.global_state["Stringency Level"][self.world.timestep])
            
            # Reduced gdp multiplier is a number calculated from Stringency Level
            self.reduced_gdp_multiplier_per_day = (1 + self.reduced_gdp_multiplier_per_year) ** (1 / 365) - 1
            self.world.global_state["Reduced GDP Multiplier"][self.world.timestep] = \
                np.mean(self.world.global_state["Stringency Level"][self.world.timestep]
                        * self.reduced_gdp_multiplier_per_day)

    def generate_observations(self):

        # Normalized observations
        obs_dict = dict()
        agent_policy_indicators = self.world.global_state["Stringency Level"][
            self.world.timestep
        ]
        # obs_dict["a"] = {
        #     "agent_policy_indicators": agent_policy_indicators
        #     / self.n_stringency_levels
        # }
        for agent in self.world.agents:
            
            
            obs_dict[agent.idx] = {
                "agent_policy_indicators": agent_policy_indicators[int(agent.idx)]
                / self.n_stringency_levels
            }
        obs_dict[self.world.planner.idx] = {
            "agent_policy_indicators": agent_policy_indicators
            / self.n_stringency_levels
        }

        return obs_dict

@component_registry.add
class FederalGovernmentSubsidyAndQuantitativePolicies(BaseComponent):
    """
    Args:
        subsidy_quantitative_policy_interval (int): The number of days over which the total subsidy amount
            is evenly rolled out.
            Note: shortening the subsidy interval increases the total amount of money
            that the planner could possibly spend. For instance, if the subsidy
            interval is 30, the planner can create a subsidy every 30 days.
        num_subsidy_quantitative_policy_level (int): The number of subsidy levels.
            Note: with max_annual_monetary_unit_per_person=10000, one round of subsidies/quantitative policy at
            the maximum level equals an expenditure of roughly $3.3 trillion
            (given the US population of 330 million).
            If the planner chooses the maximum subsidy amount, the $3.3 trillion
            is rolled out gradually over the subsidy interval.
        max_annual_monetary_unit_per_person (float): The maximum annual subsidy that may be
            allocated per person.
    """

    name = "FederalGovernmentSubsidyAndQuantitativePolicies"
    required_entities = []
    agent_subclasses = ["BasicPlanner"]

    def __init__(
        self,
        *base_component_args,
        subsidy_quantitative_policy_interval=15, #90,
        num_subsidy_quantitative_policy_level=15,
        max_annual_monetary_unit_per_person=20000,
        **base_component_kwargs,
    ):
        self.subsidy_quantitative_policy_interval = int(subsidy_quantitative_policy_interval)
        assert self.subsidy_quantitative_policy_interval >= 1

        self.num_subsidy_quantitative_policy_level = int(num_subsidy_quantitative_policy_level)
        assert self.num_subsidy_quantitative_policy_level >= 1

        self.max_annual_monetary_unit_per_person = float(max_annual_monetary_unit_per_person)
        assert self.max_annual_monetary_unit_per_person >= 0

        self.np_int_dtype = np.int32

        # (This will be overwritten during component_step; see below)
        self._subsidy_amount_per_level = None
        self._subsidy_quantitative_policy_level_array = None

        super().__init__(*base_component_args, **base_component_kwargs)

        self.default_planner_action_mask = [1 for _ in range(self.num_subsidy_quantitative_policy_level)]
        self.no_op_planner_action_mask = [0 for _ in range(self.num_subsidy_quantitative_policy_level)]

        # (This will be overwritten during reset; see below)
        self.max_daily_subsidy_per_state = np.array(
            self.n_agents, dtype=self.np_int_dtype
        )
        self.max_daily_quantitative_per_state = np.array(
            self.n_agents, dtype=self.np_int_dtype
        )

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicPlanner":
            return {"Federal Reserve Balance Sheet": np.array([0]).astype(
                np.float32
            ), "Total Subsidy": 0, "Current Subsidy Quantitative Policy Level": 0}
        return {}

    def additional_reset_steps(self):
        # Pre-compute maximum state-specific subsidy levels
        self.max_daily_subsidy_per_state = (
            self.world.us_state_population * self.max_annual_monetary_unit_per_person / 365
        )
        # the FED can set a quantitative policy with the same maximum limit as the subsidy
        self.max_daily_quantitative_per_state = (
            self.world.us_state_population * self.max_annual_monetary_unit_per_person / 365
        )

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicPlanner":
            # Number of non-zero subsidy levels
            # (the action 0 pertains to the no-subsidy case)
            return self.num_subsidy_quantitative_policy_level
        return None

    def generate_masks(self, completions=0):
        masks = {}
        if self.world.use_real_world_policies:
            masks[self.world.planner.idx] = self.default_planner_action_mask
        else:
            if self.world.timestep % self.subsidy_quantitative_policy_interval == 0:
                masks[self.world.planner.idx] = self.default_planner_action_mask
            else:
                masks[self.world.planner.idx] = self.no_op_planner_action_mask
        return masks

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        data_dict = DataFeed()
        data_dict.add_data(
            name="subsidy_quantitative_policy_interval",
            data=self.subsidy_quantitative_policy_interval,
        )
        data_dict.add_data(
            name="num_subsidy_quantitative_policy_level",
            data=self.num_subsidy_quantitative_policy_level,
        )
        data_dict.add_data(
            name="max_daily_subsidy_per_state",
            data=self.max_daily_subsidy_per_state,
        )
        data_dict.add_data(
            name="max_daily_quantitative_per_state",
            data=self.max_daily_quantitative_per_state,
        )
        data_dict.add_data(
            name="default_planner_action_mask",
            data=[1] + self.default_planner_action_mask,
        )
        data_dict.add_data(
            name="no_op_planner_action_mask",
            data=[1] + self.no_op_planner_action_mask,
        )
        return data_dict

    def get_tensor_dictionary(self):
        """
        Create a dictionary of (Pytorch-accessible) data to push to the device
        """
        tensor_dict = DataFeed()
        return tensor_dict

    def component_step(self):
        if self.world.use_cuda:
            self.world.cuda_component_step[self.name](
                self.world.cuda_data_manager.device_data("subsidy_quantitative_policy_level"),
                self.world.cuda_data_manager.device_data("subsidy"),
                self.world.cuda_data_manager.device_data("subsidy_quantitative_policy_interval"),
                self.world.cuda_data_manager.device_data("num_subsidy_quantitative_policy_level"),
                self.world.cuda_data_manager.device_data("max_daily_subsidy_per_state"),
                self.world.cuda_data_manager.device_data("max_daily_quantitative_per_state"),
                self.world.cuda_data_manager.device_data("default_planner_action_mask"),
                self.world.cuda_data_manager.device_data("no_op_planner_action_mask"),
                self.world.cuda_data_manager.device_data(f"{_ACTIONS}_p"),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_{self.name}-t_until_next_subsidy"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_{self.name}-current_subsidy_quantitative_policy_level"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_{self.name}-t_until_next_subsidy"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_{self.name}-current_subsidy_quantitative_policy_level"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_action_mask"
                ),
                self.world.cuda_data_manager.device_data("_timestep_"),
                self.world.cuda_data_manager.device_data("quantitative"),
                self.world.cuda_data_manager.meta_info("n_agents"),
                self.world.cuda_data_manager.meta_info("episode_length"),
                block=self.world.cuda_function_manager.block,
                grid=self.world.cuda_function_manager.grid,
            )
        else:
            if self.world.use_real_world_policies:
                if self._subsidy_amount_per_level is None:
                    real_life_policy_interval = self.subsidy_quantitative_policy_interval
                    self._subsidy_amount_per_level = (
                        self.world.us_population
                        * self.max_annual_monetary_unit_per_person
                        / 20
                        * real_life_policy_interval
                        / 365
                    )
                    self._subsidy_quantitative_policy_level_array = np.zeros((self._episode_length + 1))
                # Use the action taken in the previous timestep
                current_subsidy_amount = self.world.real_world_subsidy[
                    self.world.timestep - 1
                ]
                if current_subsidy_amount > 0:
                    _subsidy_quantitative_policy_level = np.round(
                        (current_subsidy_amount / self._subsidy_amount_per_level)
                    )
                    for t_idx in range(
                        self.world.timestep - 1,
                        min(
                            len(self._subsidy_quantitative_policy_level_array),
                            self.world.timestep - 1 + self.subsidy_quantitative_policy_interval,
                        ),
                    ):
                        self._subsidy_quantitative_policy_level_array[t_idx] += _subsidy_quantitative_policy_level
                subsidy_quantitative_policy_level = self._subsidy_quantitative_policy_level_array[self.world.timestep - 1]
                
                if self.world.timestep == 1:
                    self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep] = \
                            self.world.real_world_fed_fund_rate[self.world.timestep - 1][0]
                if self.world.timestep + 1 <= self._episode_length - 1:
                    self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep + 1] = \
                    self.world.real_world_fed_fund_rate[self.world.timestep][0] if int(self.world.real_world_fed_fund_rate[self.world.timestep][0]) != 0 \
                    else self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep]
                    
                    subsidy_quantitative_policy_level_frac = subsidy_quantitative_policy_level / 20
                    daily_statewise_subsidy = (
                        subsidy_quantitative_policy_level_frac * self.max_daily_subsidy_per_state
                    )

                    self.world.global_state["Subsidy"][
                        self.world.timestep
                    ] = daily_statewise_subsidy
                    self.world.planner.state["Total Subsidy"] += np.sum(daily_statewise_subsidy)

                    self.world.planner.state["Federal Reserve Balance Sheet"] = self.world.real_world_quantitative[self.world.timestep - 1][0] * 10**6  \
                        if self.world.real_world_quantitative[self.world.timestep - 1][0] != 0 else self.world.planner.state["Federal Reserve Balance Sheet"]
                    self.world.global_state["Federal Reserve Balance Sheet"] = self.world.real_world_quantitative[self.world.timestep - 1][0] * 10**6 \
                        if self.world.real_world_quantitative[self.world.timestep - 1][0] != 0 else self.world.global_state["Federal Reserve Balance Sheet"]
            else:
                # Update the subsidy level only every self.subsidy_quantitative_policy_interval, since the
                # other actions are masked out.
                if (self.world.timestep - 1) % self.subsidy_quantitative_policy_interval == 0:
                    subsidy_quantitative_policy_level = self.world.planner.get_component_action(self.name)
                else:
                    subsidy_quantitative_policy_level = self.world.planner.state["Current Subsidy Quantitative Policy Level"]

                if not (0 <= subsidy_quantitative_policy_level and subsidy_quantitative_policy_level <= self.num_subsidy_quantitative_policy_level):
                    print("subsidy_quantitative_policy_level: ", subsidy_quantitative_policy_level)
                assert 0 <= subsidy_quantitative_policy_level <= self.num_subsidy_quantitative_policy_level
                self.world.planner.state["Current Subsidy Quantitative Policy Level"] = np.array(
                    subsidy_quantitative_policy_level
                ).astype(self.np_int_dtype)
                if self.world.timestep == 0 and self.world.global_state["Federal Reserve Balance Sheet"] is not None:
                    self.world.planner.state["Federal Reserve Balance Sheet"] += self.world.global_state["Federal Reserve Balance Sheet"]
    
                # "US Tax Wedge"
                # "US Government Defense Spending",
                # "US Government Social Security Spending",
                # "US Government Medicare Medicaid Spending",
                # "US Government Non Defense Others Spending",
                # Update subsidy - quantitative easing level
                interest_hikes = 0.25
                if self.world.timestep + 1 <= self._episode_length:
                    self.world.global_state["US Government Defense Spending"][self.world.timestep + 1] \
                        = self.world.global_state["US Government Defense Spending"][self.world.timestep] 
                    
                    self.world.global_state["US Government Social Security Spending"][self.world.timestep + 1] \
                        = self.world.global_state["US Government Social Security Spending"][self.world.timestep] 
                    
                    self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep + 1] \
                        = self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep] 
                    
                    self.world.global_state["US Government Income Security"][self.world.timestep + 1] \
                        = self.world.global_state["US Government Income Security"][self.world.timestep] 
                    
                    self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep + 1] = \
                        self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep] 
                    
                    hundred_billions_divided_by_365 = 10**9 / 365
                    if subsidy_quantitative_policy_level == 0 or subsidy_quantitative_policy_level == 1: # 0 - 1
                        sign = 1 if subsidy_quantitative_policy_level == 1 else -1

                        # if rate go to 0.25, then no more reduction
                        if sign == -1 and self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep] == 0.25:
                            sign = 0
                        self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep + 1] = \
                            self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep] + sign * interest_hikes
                    elif subsidy_quantitative_policy_level == 2 or subsidy_quantitative_policy_level == 3: # 2 - 3
                        # if subsidy_quantitative_policy_level = 2, mean there is no subsidies
                        plus_or_minus = 1 if subsidy_quantitative_policy_level == 3 else 0
                        subsidy_quantitative_policy_level_frac = 0.5
                        daily_statewise_subsidy = (
                            subsidy_quantitative_policy_level_frac * self.max_daily_subsidy_per_state
                        ) * plus_or_minus
                        self.world.global_state["Subsidy"][
                                self.world.timestep
                            ] = daily_statewise_subsidy
                        self.world.planner.state["Total Subsidy"] += np.sum(daily_statewise_subsidy)
                    # quantitative easing action - only increase the self.world.global_state["Quantitative"]
                    # value where level 20 to 30 is the quantitative tightening action, from 31 to 40 is the quantitative easing action
                    elif subsidy_quantitative_policy_level == 4 \
                        or subsidy_quantitative_policy_level == 5: # 4 - 5
                        plus_or_minus = 1 if subsidy_quantitative_policy_level == 5 else -1
                        subsidy_quantitative_policy_level_frac = 0.5
                        daily_statewise_quantitative = (
                            subsidy_quantitative_policy_level_frac * self.max_daily_quantitative_per_state
                        ) * plus_or_minus

                        # self.world.global_state["Quantitative"][
                        #     self.world.timestep
                        # ] = daily_statewise_quantitative
                        if (self.world.global_state["Federal Reserve Balance Sheet"] + np.sum(daily_statewise_quantitative) < 0):
                            self.world.planner.state["Federal Reserve Balance Sheet"] += 0
                            self.world.global_state["Federal Reserve Balance Sheet"] = 0
                        else:
                            self.world.planner.state["Federal Reserve Balance Sheet"] += (self.world.global_state["Federal Reserve Balance Sheet"] + np.sum(daily_statewise_quantitative)) 
                            self.world.global_state["Federal Reserve Balance Sheet"] += np.sum(daily_statewise_quantitative)
                    elif subsidy_quantitative_policy_level == 6 \
                        or subsidy_quantitative_policy_level == 7: # 6 - 7
                        plus_or_minus = 1 if subsidy_quantitative_policy_level == 7 else -1
                        subsidy_quantitative_policy_level_frac = 1 * plus_or_minus
                        if (subsidy_quantitative_policy_level_frac < 0):
                            # Taxation cannot be lower than 10% of GDP, so the federal government cannot lower the tax if the tax wedge is 10% already
                            if self.world.global_state["US Tax Wedge"] + subsidy_quantitative_policy_level_frac * 0.1 >= 0.1:
                                self.world.global_state["US Tax Wedge"] += subsidy_quantitative_policy_level_frac * 0.1 # increasing 10% in GDP Taxation Wedge
                        else:
                            # Taxation cannot be higher than 70% of GDP as it is the realistic cap of how much the government can get
                            if self.world.global_state["US Tax Wedge"] + subsidy_quantitative_policy_level_frac * 0.1 <= 0.7:
                                self.world.global_state["US Tax Wedge"] += subsidy_quantitative_policy_level_frac * 0.1 # increasing 10% in GDP Taxation Wedge
                    
                    elif subsidy_quantitative_policy_level == 8 \
                        or subsidy_quantitative_policy_level == 9: # 8 - 9
                        plus_or_minus = 1 if subsidy_quantitative_policy_level == 9 else -1
                        subsidy_quantitative_policy_level_frac = 1 * plus_or_minus
                        self.world.global_state["US Government Defense Spending"][self.world.timestep + 1] \
                            = self.world.global_state["US Government Defense Spending"][self.world.timestep] + subsidy_quantitative_policy_level_frac * hundred_billions_divided_by_365

                    elif subsidy_quantitative_policy_level == 10 \
                        or subsidy_quantitative_policy_level == 11: # 10 - 11
                        plus_or_minus = 1 if subsidy_quantitative_policy_level == 11 else -1
                        subsidy_quantitative_policy_level_frac = plus_or_minus * 1
                        self.world.global_state["US Government Social Security Spending"][self.world.timestep + 1] \
                            = self.world.global_state["US Government Social Security Spending"][self.world.timestep] + subsidy_quantitative_policy_level_frac * hundred_billions_divided_by_365

                    elif subsidy_quantitative_policy_level == 12 \
                        or subsidy_quantitative_policy_level == 13:
                        plus_or_minus = 1 if subsidy_quantitative_policy_level == 13 else -1
                        subsidy_quantitative_policy_level_frac = 1 * plus_or_minus
                        self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep + 1] \
                            = self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep] + subsidy_quantitative_policy_level_frac * hundred_billions_divided_by_365
                    
                    elif subsidy_quantitative_policy_level == 14 \
                        or subsidy_quantitative_policy_level == 15:
                        plus_or_minus = 1 if subsidy_quantitative_policy_level == 15 else -1
                        subsidy_quantitative_policy_level_frac = 1 * plus_or_minus
                        self.world.global_state["US Government Income Security"][self.world.timestep + 1] \
                            = self.world.global_state["US Government Income Security"][self.world.timestep] + subsidy_quantitative_policy_level_frac * hundred_billions_divided_by_365
                




    def generate_observations(self):
        # Allow the agents/planner to know when the next subsidy might come.
        # Obs should = 0 when the next timestep could include a subsidy
        t_since_last_subsidy = self.world.timestep % self.subsidy_quantitative_policy_interval
        # (this is normalized to 0<-->1)
        t_until_next_subsidy = self.subsidy_quantitative_policy_interval - t_since_last_subsidy
        t_vec = t_until_next_subsidy * np.ones(self.n_agents)

        current_subsidy_quantitative_policy_level = self.world.planner.state["Current Subsidy Quantitative Policy Level"]
        sl_vec = current_subsidy_quantitative_policy_level * np.ones(self.n_agents)

        # Normalized observations
        obs_dict = dict()
        # obs_dict["a"] = {
        #     "t_until_next_subsidy": t_vec / self.subsidy_quantitative_policy_interval,
        #     "current_subsidy_quantitative_policy_level": sl_vec / self.num_subsidy_quantitative_policy_level,
        # }
        for agent in self.world.agents:
            
            obs_dict[agent.idx] = {
                "t_until_next_subsidy": t_vec[int(agent.idx)] / self.subsidy_quantitative_policy_interval,
                "current_subsidy_quantitative_policy_level": sl_vec / self.num_subsidy_quantitative_policy_level,
            }
        obs_dict[self.world.planner.idx] = {
            "t_until_next_subsidy": t_until_next_subsidy / self.subsidy_quantitative_policy_interval,
            "current_subsidy_quantitative_policy_level": current_subsidy_quantitative_policy_level / self.num_subsidy_quantitative_policy_level,
        }

        return obs_dict
 
    
@component_registry.add
class VaccinationCampaign(BaseComponent):
    """
    Implements a (passive) component for delivering vaccines to agents once a certain
    amount of time has elapsed.

    Args:
        daily_vaccines_per_million_people (int): The number of vaccines available per
            million people everyday.
        delivery_interval (int): The number of days between vaccine deliveries.
        vaccine_delivery_start_date (string): The date (YYYY-MM-DD) when the
            vaccination begins.
    """

    name = "VaccinationCampaign"
    required_entities = []
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        daily_vaccines_per_million_people=4500,
        delivery_interval=1,
        vaccine_delivery_start_date="2020-12-22",
        observe_rate=False,
        **base_component_kwargs,
    ):
        self.daily_vaccines_per_million_people = int(daily_vaccines_per_million_people)
        assert 0 <= self.daily_vaccines_per_million_people <= 1e6

        self.delivery_interval = int(delivery_interval)
        assert 1 <= self.delivery_interval <= 5000

        try:
            self.vaccine_delivery_start_date = datetime.strptime(
                vaccine_delivery_start_date, "%Y-%m-%d"
            )
        except ValueError:
            print("Incorrect data format, should be YYYY-MM-DD")

        # (This will  be overwritten during component_step (see below))
        self._time_when_vaccine_delivery_begins = None

        self.np_int_dtype = np.int32

        self.observe_rate = bool(observe_rate)

        super().__init__(*base_component_args, **base_component_kwargs)

        # (This will be overwritten during reset; see below)
        self._num_vaccines_per_delivery = None
        # Convenience for obs (see usage below):
        self._t_first_delivery = None

    @property
    def num_vaccines_per_delivery(self):
        if self._num_vaccines_per_delivery is None:
            # Pre-compute dispersal numbers
            millions_of_residents = self.world.us_state_population / 1e6
            daily_vaccines = (
                millions_of_residents * self.daily_vaccines_per_million_people
            )
            num_vaccines_per_delivery = np.floor(
                self.delivery_interval * daily_vaccines
            )
            self._num_vaccines_per_delivery = np.array(
                num_vaccines_per_delivery, dtype=self.np_int_dtype
            )
        return self._num_vaccines_per_delivery

    @property
    def time_when_vaccine_delivery_begins(self):
        if self._time_when_vaccine_delivery_begins is None:
            self._time_when_vaccine_delivery_begins = (
                self.vaccine_delivery_start_date - self.world.start_date
            ).days
        return self._time_when_vaccine_delivery_begins

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return {"Total Vaccinated": 0, "Vaccines Available": 0}
        return {}

    def additional_reset_steps(self):
        pass

    def get_n_actions(self, agent_cls_name):
        return  # Passive component

    def generate_masks(self, completions=0):
        return {}  # Passive component

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        data_dict = DataFeed()
        data_dict.add_data(
            name="num_vaccines_per_delivery",
            data=self.num_vaccines_per_delivery,
        )
        data_dict.add_data(
            name="delivery_interval",
            data=self.delivery_interval,
        )
        data_dict.add_data(
            name="time_when_vaccine_delivery_begins",
            data=self.time_when_vaccine_delivery_begins,
        )
        data_dict.add_data(
            name="num_vaccines_available_t",
            data=np.zeros(self.n_agents),
            save_copy_and_apply_at_reset=True,
        )
        return data_dict

    def get_tensor_dictionary(self):
        """
        Create a dictionary of (Pytorch-accessible) data to push to the device
        """
        tensor_dict = DataFeed()
        return tensor_dict

    def component_step(self):
        if self.world.use_cuda:
            self.world.cuda_component_step[self.name](
                self.world.cuda_data_manager.device_data("vaccinated"),
                self.world.cuda_data_manager.device_data("num_vaccines_per_delivery"),
                self.world.cuda_data_manager.device_data("num_vaccines_available_t"),
                self.world.cuda_data_manager.device_data("delivery_interval"),
                self.world.cuda_data_manager.device_data(
                    "time_when_vaccine_delivery_begins"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_{self.name}-t_until_next_vaccines"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_{self.name}-t_until_next_vaccines"
                ),
                self.world.cuda_data_manager.device_data("_timestep_"),
                self.world.cuda_data_manager.meta_info("n_agents"),
                self.world.cuda_data_manager.meta_info("episode_length"),
                block=self.world.cuda_function_manager.block,
                grid=self.world.cuda_function_manager.grid,
            )
        else:
            # Do nothing if vaccines are not available yet
            if self.world.timestep < self.time_when_vaccine_delivery_begins:
                return

            # Do nothing if this is not the start of a delivery interval.
            # Vaccines are delivered at the start of each interval.
            if (self.world.timestep % self.delivery_interval) != 0:
                return
            # Deliver vaccines to each state
            # total_vaccine = 0
            # for aidx, agent in enumerate(self.world.agents):  
            #     total_vaccine += agent.state["Total Vaccinated"]
            # US can only get 68% of their population to get the vaccine
            # if total_vaccine / self.world.us_population * 100 > 68:
            #     return 
            for aidx, vaccines in enumerate(self.num_vaccines_per_delivery):
                self.world.agents[aidx].state["Vaccines Available"] += vaccines

    def generate_observations(self):
        # Allow the agents/planner to know when the next vaccines might come.
        # Obs should = 0 when the next timestep will deliver vaccines
        # (this is normalized to 0<-->1)

        if self._t_first_delivery is None:
            self._t_first_delivery = int(self.time_when_vaccine_delivery_begins)
            while (self._t_first_delivery % self.delivery_interval) != 0:
                self._t_first_delivery += 1

        next_t = self.world.timestep + 1
        if next_t <= self._t_first_delivery:
            t_until_next_vac = np.minimum(
                1, (self._t_first_delivery - next_t) / self.delivery_interval
            )
            next_vax_rate = 0.0
        else:
            t_since_last_vac = next_t % self.delivery_interval
            t_until_next_vac = self.delivery_interval - t_since_last_vac
            next_vax_rate = self.daily_vaccines_per_million_people / 1e6
        t_vec = t_until_next_vac * np.ones(self.n_agents)
        r_vec = next_vax_rate * np.ones(self.n_agents)

        # Normalized observations
        obs_dict = dict()
        # obs_dict["a"] = {"t_until_next_vaccines": t_vec / self.delivery_interval}
        for agent in self.world.agents: 
            
            obs_dict[agent.idx] = {"t_until_next_vaccines": t_vec[int(agent.idx)] / self.delivery_interval}
        obs_dict[self.world.planner.idx] = {
            "t_until_next_vaccines": t_until_next_vac / self.delivery_interval
        }

        if self.observe_rate:
            # obs_dict["a"]["next_vaccination_rate"] = r_vec
            
            for agent in self.world.agents:
                # if(agent.idx == 50): 
                #     continue
                obs_dict[agent.idx]["next_vaccination_rate"] = r_vec[int(agent.idx)]
            obs_dict["p"]["next_vaccination_rate"] = float(next_vax_rate)

        return obs_dict
