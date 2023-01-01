import ai_economist
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from datetime import timedelta
from timeit import Timer

from ai_economist.foundation.scenarios.covid19.covid19_env import (
    CovidAndEconomyEnvironment,
)
from ai_economist.foundation.env_wrapper import FoundationEnvWrapper 

from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.training.trainer import Trainer
from warp_drive.training.utils.data_loader import create_and_push_data_placeholders
from warp_drive.utils.env_registrar import EnvironmentRegistrar

_PATH_TO_AI_ECONOMIST_PACKAGE_DIR = ai_economist.__path__[0]

# Set font size for the matplotlib figures
plt.rcParams.update({'font.size': 22}) # Set logger level e.g., DEBUG, INFO, WARNING, ERROR
import logging

logging.getLogger().setLevel(logging.DEBUG)

# Ensure that a GPU is present.
import GPUtil
num_gpus_available = len(GPUtil.getAvailable())
assert num_gpus_available > 0, "This notebook needs a GPU machine to run!!"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

env_config = {
    'collate_agent_step_and_reset_data': True,
     'components': [
         {'ControlUSStateOpenCloseStatus': {'action_cooldown_period': 28}},
          {'FederalGovernmentSubsidy': {'num_subsidy_levels': 20,
            'subsidy_interval': 90,
            'max_annual_subsidy_per_person': 20000}},
          {'VaccinationCampaign': {'daily_vaccines_per_million_people': 3000,
            'delivery_interval': 1,
            'vaccine_delivery_start_date': '2021-01-12'}},
        {'FederalQuantitativeEasing': {
            # The number of QE levels.
            'num_QE_levels': 20,
            # The number of days over which the total subsidy amount is evenly rolled out.
            'QE_interval': 90,
            # The maximum annual subsidy that may be allocated per person.
            'max_annual_QE_per_person': 20
        }},
     ],
     'economic_reward_crra_eta': 2,
     'episode_length': 540,
     'flatten_masks': True,
     'flatten_observations': False,
     'health_priority_scaling_agents': 0.3,
     'health_priority_scaling_planner': 0.45,
     'infection_too_sick_to_work_rate': 0.1,
     'multi_action_mode_agents': False,
     'multi_action_mode_planner': False,
     'n_agents': 51,
     'path_to_data_and_fitted_params': '',
     'pop_between_age_18_65': 0.6,
     'risk_free_interest_rate': 0.03,
     'world_size': [1, 1],
     'start_date': '2020-03-22',
     'use_real_world_data': False,
     'use_real_world_policies': False
}



env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(
    CovidAndEconomyEnvironment.name,
    os.path.join(
        _PATH_TO_AI_ECONOMIST_PACKAGE_DIR, 
        "foundation/scenarios/covid19/covid19_build.cu"
    )
)



# The policy_tag_to_agent_id_map dictionary maps
# policy model names to agent ids.
policy_tag_to_agent_id_map = {
    "a": [str(agent_id) for agent_id in range(env_config["n_agents"])],
    "p": ["p"],
    "f": ["f"]
}

# Flag indicating whether separate obs, actions and rewards placeholders have to be created for each policy.
# Set "create_separate_placeholders_for_each_policy" to True here 
# since the agents and the planner have different observation and action spaces.
separate_placeholder_per_policy = True



# Flag indicating the observation dimension corresponding to 'num_agents'.
# Note: WarpDrive assumes that all the observation are shaped
# (num_agents, *feature_dim), i.e., the observation dimension
# corresponding to 'num_agents' is the first one. Instead, if the
# observation dimension corresponding to num_agents is the last one,
# we will need to permute the axes to align with WarpDrive's assumption
obs_dim_corresponding_to_num_agents = "last"

config_path = os.path.join(
    _PATH_TO_AI_ECONOMIST_PACKAGE_DIR,
    "training/run_configs/",
    f"covid_and_economy_environment.yaml",
)
with open(config_path, "r", encoding="utf8") as f:
    run_config = yaml.safe_load(f)



if __name__ == "__main__":  
  EnvironmentCPUvsGPU(
      dual_mode_env_class=CovidAndEconomyEnvironment,
      env_configs={"test": env_config},
      num_envs=1,
      blocks_per_env=3,
      num_episodes=100,
      env_wrapper=FoundationEnvWrapper,
      env_registrar=env_registrar,
      policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
      create_separate_placeholders_for_each_policy=True,
      obs_dim_corresponding_to_num_agents="last",  
  ).test_env_reset_and_step()