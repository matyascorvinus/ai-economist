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


# @component_registry.add
# class FederalInterestRate(BaseComponent): 

#     name = "FederalInterestRate"
#     required_entities = []
#     agent_subclasses = ["BasicPlanner", "BasicFederalReserve"]

#     def __init__(
#         self,
#         interest_rate_intervals=90,
#         num_interest_rate_levels=40,  # numbers from 0 - 40 - Highest interest rate should be 20%
#         *base_component_args, 
#         **base_component_kwargs,
#     ):
         
#         self.np_int_dtype = np.int32 
#         self.num_interest_rate_levels = num_interest_rate_levels
#         self.interest_rate_intervals = interest_rate_intervals
#         super().__init__(*base_component_args, **base_component_kwargs)
#         # (This will be overwritten during reset; see below) 
#         self.fed_interest_rate = 0.0

#     def get_additional_state_fields(self, agent_cls_name):
#         if agent_cls_name == "BasicFederalReserve":
#             return {"Interest Rate": 0, "Current Interest Rate Level": 0}
#         return {}

#     def additional_reset_steps(self):
#         # Pre-compute maximum state-specific Interest Rate levels
#         self.fed_interest_rate = self.world.global_state["Interest Rate"];

#     def get_n_actions(self, agent_cls_name):
#         if agent_cls_name == "BasicFederalReserve":
#             # Number of non-zero QE levels
#             # (the action 0 pertains to the no-QE case)
#             return self.num_interest_rate_levels
#         return None

#     def generate_masks(self, completions=0):
#         masks = {}
#         if self.world.use_real_world_policies:
#             masks[self.world.federal_reserve.idx] = self.default_federal_reserve_action_mask
#         else:
#             if self.world.timestep % self.interest_rate_intervals == 0:
#                 masks[self.world.federal_reserve.idx] = self.default_federal_reserve_action_mask
#             else:
#                 masks[self.world.federal_reserve.idx] = self.no_op_federal_reserve_action_mask
#         return masks

#     def get_data_dictionary(self):
#         """
#         Create a dictionary of data to push to the device
#         """
#         data_dict = DataFeed()
#         data_dict.add_data(
#             name="Fed Interest Rate",
#             data=self.fed_interest_rate,
#         )
         
#         return data_dict

#     def get_tensor_dictionary(self):
#         """
#         Create a dictionary of (Pytorch-accessible) data to push to the device
#         """
#         tensor_dict = DataFeed()
#         return tensor_dict

#     def component_step(self):
#         if self.world.use_cuda:
#             self.world.cuda_component_step[self.name](
#                 self.world.cuda_data_manager.device_data("QE_level"),
#                 self.world.cuda_data_manager.device_data("QE"),
#                 self.world.cuda_data_manager.device_data("QE_interval"),
#                 self.world.cuda_data_manager.device_data("num_QE_levels"), 
#                 self.world.cuda_data_manager.device_data("default_federal_reserve_action_mask"),
#                 self.world.cuda_data_manager.device_data("no_op_federal_reserve_action_mask"),
#                 self.world.cuda_data_manager.device_data(f"{_ACTIONS}_p"),
#                 self.world.cuda_data_manager.device_data(
#                     f"{_OBSERVATIONS}_a_{self.name}-t_until_next_QE"
#                 ),
#                 self.world.cuda_data_manager.device_data(
#                     f"{_OBSERVATIONS}_a_{self.name}-current_QE_level"
#                 ),
#                 self.world.cuda_data_manager.device_data(
#                     f"{_OBSERVATIONS}_p_{self.name}-t_until_next_QE"
#                 ),
#                 self.world.cuda_data_manager.device_data(
#                     f"{_OBSERVATIONS}_p_{self.name}-current_QE_level"
#                 ),
#                 self.world.cuda_data_manager.device_data(
#                     f"{_OBSERVATIONS}_p_action_mask"
#                 ),
#                 self.world.cuda_data_manager.device_data("_timestep_"),
#                 self.world.cuda_data_manager.meta_info("n_agents"),
#                 self.world.cuda_data_manager.meta_info("episode_length"),
#                 block=self.world.cuda_function_manager.block,
#                 grid=self.world.cuda_function_manager.grid,
#             )
#         else:
#             if self.world.use_real_world_policies:
#                 if self._QE_amount_per_level is None:
#                     self._QE_amount_per_level = (
#                         self.world.us_population
#                         * self.max_annual_QE_per_person
#                         / self.num_QE_levels
#                         * self.QE_interval
#                         / 365
#                     )
#                     self._QE_level_array = np.zeros((self._episode_length + 1))
#                 # Use the action taken in the previous timestep
#                 current_QE_amount = self.world.real_world_QE[
#                     self.world.timestep - 1
#                 ]
#                 if current_QE_amount > 0:
#                     _QE_level = np.round(
#                         (current_QE_amount / self._QE_amount_per_level)
#                     )
#                     for t_idx in range(
#                         self.world.timestep - 1,
#                         min(
#                             len(self._QE_level_array),
#                             self.world.timestep - 1 + self.QE_interval,
#                         ),
#                     ):
#                         self._QE_level_array[t_idx] += _QE_level
#                 QE_level = self._QE_level_array[self.world.timestep - 1]
#             else:
#                 Update the QE level only every self.QE_interval, since the
#                 other actions are masked out.
#             if (self.world.timestep - 1) % self.QE_interval == 0:
#                 QE_level = self.world.federal_reserve.get_component_action(self.name)
#             else:
#                 QE_level = self.world.federal_reserve.state["Current QE Level"]

#             assert 0 <= QE_level <= self.num_QE_levels
#             self.world.federal_reserve.state["Current QE Level"] = np.array(
#                 QE_level
#             ).astype(self.np_int_dtype)

#             # Update QE level
#             QE_level_frac = QE_level / self.num_QE_levels
#             daily_statewise_QE = (
#                 QE_level_frac * self.max_daily_QE_per_state
#             )

#             self.world.global_state["QE"][
#                 self.world.timestep
#             ] = daily_statewise_QE
#             self.world.federal_reserve.state["Total QE"] += np.sum(daily_statewise_QE)

#     def generate_observations(self):
#         # Allow the agents/federalreserve to know when the next QE might come.
#         # Obs should = 0 when the next timestep could include a QE
#         t_since_last_QE = self.world.timestep % self.QE_interval
#         # (this is normalized to 0<-->1)
#         t_until_next_QE = self.QE_interval - t_since_last_QE
#         t_vec = t_until_next_QE * np.ones(self.n_agents)

#         current_QE_level = self.world.federal_reserve.state["Current QE Level"]
#         sl_vec = current_QE_level * np.ones(self.n_agents)

#         # Normalized observations
#         obs_dict = dict()
#         obs_dict["f"] = {
#             "t_until_next_QE": t_vec / self.QE_interval,
#             "current_QE_level": sl_vec / self.num_QE_levels,
#         }
        
#         obs_dict[self.world.federal_reserve.idx] = {
#             "t_until_next_QE": t_until_next_QE / self.QE_interval,
#             "current_QE_level": current_QE_level / self.num_QE_levels,
#         } 

#         return obs_dict
    
@component_registry.add
class FederalQuantitativeEasing(BaseComponent):
    """
    Args:
        QE_interval (int): The number of days over which the total QE amount
            is evenly rolled out.
            Note: shortening the QE interval increases the total amount of money
            that the Federal Reserve could possibly execute Quantitative Easing. For instance, if the QE
            interval is 30, the Federal Reserve can create a QE every 30 days.
        num_QE_levels (int): The number of QE levels.
            Note: The Federal Reserve can execute a quantiative easing in level of 1 trillion USD permonth
        max_annual_QE (float): The maximum annual QE amount.
    """

    name = "FederalQuantitativeEasing"
    required_entities = []
    agent_subclasses = ["BasicPlanner", "BasicFederalReserve"]

    def __init__(
        self,
        *base_component_args,
        QE_interval=90,
        num_QE_levels=40,  # numbers from 0 - 19 represent quantitative tightening, 20 - 39 quantitative easing
        max_annual_QE_per_person=5000,
        **base_component_kwargs,
    ):
        self.QE_interval = int(QE_interval)
        assert self.QE_interval >= 1

        self.num_QE_levels = int(num_QE_levels) 
        assert self.num_QE_levels >= 1

        self.max_annual_QE_per_person = float(max_annual_QE_per_person)
        assert self.max_annual_QE_per_person >= 0

        self.np_int_dtype = np.int32

        # (This will be overwritten during component_step; see below)
        self._QE_amount_per_level = None
        self._QE_level_array = None

        super().__init__(*base_component_args, **base_component_kwargs)

        self.default_federal_reserve_action_mask = [1 for _ in range(self.num_QE_levels)]
        self.no_op_federal_reserve_action_mask = [0 for _ in range(self.num_QE_levels)]

        # (This will be overwritten during reset; see below)
        self.max_daily_QE_per_state = np.array(
            self.n_agents, dtype=self.np_int_dtype
        )

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicFederalReserve":
            return {"Total QE": 0, "Current QE Level": 0}
        return {}

    def additional_reset_steps(self):
        # Pre-compute maximum state-specific QE levels
        self.max_daily_QE_per_state = (
            self.world.us_state_population * self.max_annual_QE_per_person / 365
        )

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicFederalReserve":
            # Number of non-zero QE levels
            # (the action 0 pertains to the no-QE case)
            return self.num_QE_levels
        return None

    def generate_masks(self, completions=0):
        masks = {}
        if self.world.use_real_world_policies:
            masks[self.world.federal_reserve.idx] = self.default_federal_reserve_action_mask
        else:
            if self.world.timestep % self.QE_interval == 0:
                masks[self.world.federal_reserve.idx] = self.default_federal_reserve_action_mask
            else:
                masks[self.world.federal_reserve.idx] = self.no_op_federal_reserve_action_mask
        return masks

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        data_dict = DataFeed()
        data_dict.add_data(
            name="QE_interval",
            data=self.QE_interval,
        )
        data_dict.add_data(
            name="num_QE_levels",
            data=self.num_QE_levels,
        ) 
        data_dict.add_data(
            name="max_daily_QE_per_state",
            data=self.max_daily_QE_per_state,
        ) 
        data_dict.add_data(
            name="default_federal_reserve_action_mask",
            data=[1] + self.default_federal_reserve_action_mask,
        )
        data_dict.add_data(
            name="no_op_federal_reserve_action_mask",
            data=[1] + self.no_op_federal_reserve_action_mask,
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
                self.world.cuda_data_manager.device_data("QE_level"),
                self.world.cuda_data_manager.device_data("QE"),
                self.world.cuda_data_manager.device_data("QE_interval"),
                self.world.cuda_data_manager.device_data("num_QE_levels"),
                self.world.cuda_data_manager.device_data("max_daily_QE_per_state"),
                self.world.cuda_data_manager.device_data("default_federal_reserve_action_mask"),
                self.world.cuda_data_manager.device_data("no_op_federal_reserve_action_mask"),
                self.world.cuda_data_manager.device_data(f"{_ACTIONS}_p"),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_{self.name}-t_until_next_QE"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_{self.name}-current_QE_level"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_{self.name}-t_until_next_QE"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_{self.name}-current_QE_level"
                ),
                self.world.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_action_mask"
                ),
                self.world.cuda_data_manager.device_data("_timestep_"),
                self.world.cuda_data_manager.meta_info("n_agents"),
                self.world.cuda_data_manager.meta_info("episode_length"),
                block=self.world.cuda_function_manager.block,
                grid=self.world.cuda_function_manager.grid,
            )
        else:
            if self.world.use_real_world_policies:
                if self._QE_amount_per_level is None:
                    self._QE_amount_per_level = (
                        self.world.us_population
                        * self.max_annual_QE_per_person
                        / self.num_QE_levels
                        * self.QE_interval
                        / 365
                    )
                    self._QE_level_array = np.zeros((self._episode_length + 1))
                # Use the action taken in the previous timestep
                current_QE_amount = self.world.real_world_QE[
                    self.world.timestep - 1
                ]
                if current_QE_amount > 0:
                    _QE_level = np.round(
                        (current_QE_amount / self._QE_amount_per_level)
                    )
                    for t_idx in range(
                        self.world.timestep - 1,
                        min(
                            len(self._QE_level_array),
                            self.world.timestep - 1 + self.QE_interval,
                        ),
                    ):
                        self._QE_level_array[t_idx] += _QE_level
                QE_level = self._QE_level_array[self.world.timestep - 1]
            else:
                # Update the QE level only every self.QE_interval, since the
                # other actions are masked out.
                if (self.world.timestep - 1) % self.QE_interval == 0:
                    QE_level = self.world.federal_reserve.get_component_action(self.name)
                else:
                    QE_level = self.world.federal_reserve.state["Current QE Level"]

            assert 0 <= QE_level <= self.num_QE_levels
            self.world.federal_reserve.state["Current QE Level"] = np.array(
                QE_level
            ).astype(self.np_int_dtype)
            
            QE_level_frac = 0.0

            # Update QE level
            QE_level_frac = QE_level / (self.num_QE_levels - 20)
                
            daily_statewise_QE = (
                QE_level_frac * self.max_daily_QE_per_state
            )

            self.world.global_state["QE"][
                self.world.timestep
            ] = daily_statewise_QE
            self.world.federal_reserve.state["Money Supply"] += np.sum(daily_statewise_QE)
            self.world.global_state["Money Supply"] += np.sum(daily_statewise_QE)
            
            self.world.federal_reserve.state["FED Balance Sheet"] += np.sum(daily_statewise_QE)
            self.world.global_state["FED Balance Sheet"] += np.sum(daily_statewise_QE)
            

    def generate_observations(self):
        # Allow the agents/federalreserve to know when the next QE might come.
        # Obs should = 0 when the next timestep could include a QE
        t_since_last_QE = self.world.timestep % self.QE_interval
        # (this is normalized to 0<-->1)
        t_until_next_QE = self.QE_interval - t_since_last_QE
        t_vec = t_until_next_QE * np.ones(self.n_agents)

        current_QE_level = self.world.federal_reserve.state["Current QE Level"]
        sl_vec = current_QE_level * np.ones(self.n_agents)

        # Normalized observations
        obs_dict = dict()
        obs_dict["p"] = {
            "t_until_next_QE": t_vec / self.QE_interval,
            "current_QE_level": sl_vec / self.num_QE_levels,
        }
        
        obs_dict[self.world.federal_reserve.idx] = {
            "t_until_next_QE": t_until_next_QE / self.QE_interval,
            "current_QE_level": current_QE_level / self.num_QE_levels,
        } 

        return obs_dict
    