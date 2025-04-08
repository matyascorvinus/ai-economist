
import datetime
now = datetime.datetime.now()
name_file = 'simulation_results-AI_' + str(now).replace(':', '-').replace(' ', '_') +  '.csv'
path_to_data_and_fitted_params = "../../../datasets/covid19_datasets/2024-02-29"
env_config_dict = {
    # Scenario name - determines which scenario class to use
    "scenario_name": "CovidAndEconomySimulation",

    # The list of components in this simulation
    "components": [
        {"ControlUSStateOpenCloseStatus": {
            # action cooldown period in days.
            # Once a stringency level is set, the state(s) cannot switch to another level
            # for a certain number of days (referred to as the "action_cooldown_period")
            "action_cooldown_period": 28
        }},
        {"FederalGovernmentSubsidyAndQuantitativePolicies": {
            # The number of subsidy levels.
            "num_subsidy_quantitative_policy_level": 15,
            # The number of days over which the total subsidy amount is evenly rolled out.
            "subsidy_quantitative_policy_interval": 1,
            # The maximum annual subsidy that may be allocated per person.
            "max_annual_monetary_unit_per_person": 20000,
        }},
        {"VaccinationCampaign": {
            # The number of vaccines available per million people everyday.
            "daily_vaccines_per_million_people": 3000,
            # The number of days between vaccine deliveries.
            "delivery_interval": 1,
            # The date (YYYY-MM-DD) when vaccination begins
            "vaccine_delivery_start_date": "2021-01-12",
        }},
    ],

    # Date (YYYY-MM-DD) to start the simulation.
    "start_date": "2020-03-22",
    # How long to run the simulation for (in days)
    "episode_length": 405,

    # use_real_world_data (bool): Replay what happened in the real world.
    # Real-world data comprises SIR (susceptible/infected/recovered),
    # unemployment, government policy, and vaccination numbers.
    # This setting also sets use_real_world_policies=True.
    "use_real_world_data": False,
    # use_real_world_policies (bool): Run the environment with real-world policies
    # (stringency levels and subsidies). With this setting and
    # use_real_world_data=False, SIR and economy dynamics are still
    # driven by fitted models.
    "use_real_world_policies": False,
    "csv_validation": False,

    # A factor indicating how much more the
    # states prioritize health (roughly speaking, loss of lives due to
    # opening up more) over the economy (roughly speaking, a loss in GDP
    # due to shutting down resulting in more unemployment) compared to the
    # real-world.
    # For example, a value of 1 corresponds to the health weight that
    # maximizes social welfare under the real-world policy, while
    # a value of 2 means that states care twice as much about public health
    # (preventing deaths), while a value of 0.5 means that states care twice
    # as much about the economy (preventing GDP drops).
    "health_priority_scaling_agents": 1,
    # Same as above for the planner
    "health_priority_scaling_planner": 1,

    # Full path to the directory containing
    # the data, fitted parameters and model constants. This defaults to
    # "ai_economist/datasets/covid19_datasets/data_and_fitted_params".
    # For details on obtaining these parameters, please see the notebook
    # "ai-economist-foundation/ai_economist/datasets/covid19_datasets/
    # gather_real_world_data_and_fit_parameters.ipynb".
    "path_to_data_and_fitted_params": path_to_data_and_fitted_params,
    "us_government_spending_economic_multiplier": 1.00,

    # Economy-related parameters
    # Fraction of people infected with COVID-19. Infected people don't work.
    "infection_too_sick_to_work_rate": 0.1,
    # Fraction of the population between ages 18-65.
    # This is the subset of the population whose employment/unemployment affects
    # economic productivity.
    "pop_between_age_18_65": 0.6,
    # Percentage of interest paid by the federal
    # government to borrow money from the federal reserve for COVID-19 relief
    # (direct payments). Higher interest rates mean that direct payments
    # have a larger cost on the federal government's economic index.
    "risk_free_interest_rate": 0.03,
    # CRRA eta parameter for modeling the economic reward non-linearity.
    "economic_reward_crra_eta": 2,

    # Number of agents in the simulation (50 US states + Washington DC)
    "n_agents": 51,
    # World size: Not relevant to this simulation, but needs to be set for Foundation
    "world_size": [1, 1],
    # Flag to collate all the agents' observations, rewards and done flags into a single matrix
    "collate_agent_step_and_reset_data": False,
}
from rllib.env_wrapper import RLlibEnvWrapper
env_obj = RLlibEnvWrapper({"env_config_dict": env_config_dict})
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
policies = {
    "a": (
        None,  # uses default policy
        env_obj.observation_space,
        env_obj.action_space,
        {}  # define a custom agent policy configuration.
    ),
    "p": (
        None,  # uses default policy
        env_obj.observation_space_pl,
        env_obj.action_space_pl,
        {}  # define a custom planner policy configuration.
    )
}

# In foundation, all the agents have integer ids and the social planner has an id of "p"
policy_mapping_fun = lambda i: "a" if str(i).isdigit() else "p"

policies_to_train = ["a", "p"]

trainer_config = {
    "multiagent": {
        "policies": policies,
        "policies_to_train": policies_to_train,
        "policy_mapping_fn": policy_mapping_fun,
    },
    "num_gpus": 1, 
    #"num_gpus_per_worker": 1
    # "log_level": "DEBUG",  # Set the log level
}

trainer_config.update(
    {
        "num_workers": 1,
        "num_envs_per_worker": 1,
        # Other training parameters
        # "train_batch_size":  10,
        # "sgd_minibatch_size": 5,
        # "num_sgd_iter": 1
    }
)

# We also add the "num_envs_per_worker" parameter for the env. wrapper to index the environments.
env_config = {
    "env_config_dict": env_config_dict,
    "num_envs_per_worker": trainer_config.get('num_envs_per_worker'),
}

trainer_config.update(
    {
        "env_config": env_config
    }
)

# Initialize Ray
ray.init(webui_host="127.0.0.1")
# Create the PPO trainer.
trainer = PPOTrainer(
    env=RLlibEnvWrapper,
    config=trainer_config,
 
)

# Number of US states: 51
# self.us_gdp_2019:  21466585914800.0
# self.us_federal_deficit:  2465753424.6575336
# self.us_government_revenue:  9589041095.890411
# self.us_government_mandatory_and_discretionary_spending:  12054794520.547945
NUM_ITERS = 2
checkpoint_path = ''
checkpoint_list = []
for iteration in range(NUM_ITERS):
    print(f'********** Iter : {iteration} **********')
    result = trainer.train()
    print(f'''episode_reward_mean: {result.get('episode_reward_mean')}''')
    print(f'''episode_reward_max: {result.get('episode_reward_max')}''')
    checkpoint_path = trainer.save()
    checkpoint_list.append({
        'path': checkpoint_path, 
        'episode_reward_max': result.get('episode_reward_max'), 
        'episode_reward_mean': result.get('episode_reward_mean'), 
    })
    print("Model checkpoint saved at:", checkpoint_path)

max_reward = max(checkpoint_list, key=lambda x: x['episode_reward_mean'])

print("Path:", max_reward['path'])
print("Max Episode Reward:", max_reward['episode_reward_max'])
print("Mean Episode Reward:", max_reward['episode_reward_mean'])

checkpoint_path = checkpoint_path if (max_reward is None or max_reward['path'] is None) else max_reward['path']

import ai_economist
env_config_dict = {
    # Scenario name - determines which scenario class to use
    "scenario_name": "CovidAndEconomySimulation",

    # The list of components in this simulation
    "components": [
        {"ControlUSStateOpenCloseStatus": {
            # action cooldown period in days.
            # Once a stringency level is set, the state(s) cannot switch to another level
            # for a certain number of days (referred to as the "action_cooldown_period")
            "action_cooldown_period": 28
        }},
        {"FederalGovernmentSubsidyAndQuantitativePolicies": {
            # The number of subsidy levels.
            "num_subsidy_quantitative_policy_level": 15,
            # The number of days over which the total subsidy amount is evenly rolled out.
            "subsidy_quantitative_policy_interval": 1,
            # The maximum annual subsidy that may be allocated per person.
            "max_annual_monetary_unit_per_person": 20000,
        }},
        {"VaccinationCampaign": {
            # The number of vaccines available per million people everyday.
            "daily_vaccines_per_million_people": 3000,
            # The number of days between vaccine deliveries.
            "delivery_interval": 1,
            # The date (YYYY-MM-DD) when vaccination begins
            "vaccine_delivery_start_date": "2021-01-12",
        }},
    ],

    # Date (YYYY-MM-DD) to start the simulation.
    "start_date": "2020-03-22",
    # How long to run the simulation for (in days)
    "episode_length": 500, # 1014 days From 2020-03-22 to 2022-12-31

    # use_real_world_data (bool): Replay what happened in the real world.
    # Real-world data comprises SIR (susceptible/infected/recovered),
    # unemployment, government policy, and vaccination numbers.
    # This setting also sets use_real_world_policies=True.
    "use_real_world_data": False,
    # use_real_world_policies (bool): Run the environment with real-world policies
    # (stringency levels and subsidies). With this setting and
    # use_real_world_data=False, SIR and economy dynamics are still
    # driven by fitted models.
    "use_real_world_policies": False,
    "csv_validation": True,


    # A factor indicating how much more the
    # states prioritize health (roughly speaking, loss of lives due to
    # opening up more) over the economy (roughly speaking, a loss in GDP
    # due to shutting down resulting in more unemployment) compared to the
    # real-world.
    # For example, a value of 1 corresponds to the health weight that
    # maximizes social welfare under the real-world policy, while
    # a value of 2 means that states care twice as much about public health
    # (preventing deaths), while a value of 0.5 means that states care twice
    # as much about the economy (preventing GDP drops).
    "health_priority_scaling_agents": 1,
    # Same as above for the planner
    "health_priority_scaling_planner": 1,

    # Full path to the directory containing
    # the data, fitted parameters and model constants. This defaults to
    # "ai_economist/datasets/covid19_datasets/data_and_fitted_params".
    # For details on obtaining these parameters, please see the notebook
    # "ai-economist-foundation/ai_economist/datasets/covid19_datasets/
    # gather_real_world_data_and_fit_parameters.ipynb".
    "path_to_data_and_fitted_params": path_to_data_and_fitted_params,
    "us_government_spending_economic_multiplier": 1.00,

    # Economy-related parameters
    # Fraction of people infected with COVID-19. Infected people don't work.
    "infection_too_sick_to_work_rate": 0.1,
    # Fraction of the population between ages 18-65.
    # This is the subset of the population whose employment/unemployment affects
    # economic productivity.
    "pop_between_age_18_65": 0.6,
    # Percentage of interest paid by the federal
    # government to borrow money from the federal reserve for COVID-19 relief
    # (direct payments). Higher interest rates mean that direct payments
    # have a larger cost on the federal government's economic index.
    "risk_free_interest_rate": 0.03,
    # CRRA eta parameter for modeling the economic reward non-linearity.
    "economic_reward_crra_eta": 2,

    # Number of agents in the simulation (50 US states + Washington DC)
    "n_agents": 51,
    # World size: Not relevant to this simulation, but needs to be set for Foundation
    "world_size": [1, 1],
    # Flag to collate all the agents' observations, rewards and done flags into a single matrix
    "collate_agent_step_and_reset_data": False,
    "csv_file_path": name_file
} 
trainer.restore(checkpoint_path)
calibrated_env = RLlibEnvWrapper({"env_config_dict": env_config_dict})

DATE_FORMAT = "%Y-%m-%d"
obs = calibrated_env.reset();
# print(calibrated_env.us_state_idx_to_state_name.items())
for _ in range(calibrated_env.episode_length):
    # Set initial states
    agent_states = {}
    for agent_idx in range(env_obj.env.n_agents):
        agent_states[str(agent_idx)] = trainer.get_policy("a").get_initial_state()
    planner_states = trainer.get_policy("p").get_initial_state()   

    actions = {}
    for agent_idx in range(env_obj.env.n_agents):
        # Use the trainer object directly to sample actions for each agent
        actions[str(agent_idx)] = trainer.compute_action(
            obs[str(agent_idx)], 
            agent_states[str(agent_idx)], 
            policy_id="a",
            full_fetch=False
        )

    # Action sampling for the planner
    actions["p"] = trainer.compute_action(
        obs['p'], 
        planner_states, 
        policy_id='p',
        full_fetch=False
    )
    obs, rew, done, info = env_obj.step(actions)        
