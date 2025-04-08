# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import json
import os
from datetime import datetime, timedelta
import csv
import GPUtil
import numpy as np

from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
from ai_economist.foundation.utils import verify_activation_code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
from scipy.optimize import fsolve

headers = [
    "Month", "Susceptibles", "Infected", "Recovered", "Vaccinated (% of population)", "Deaths (thousands)" ,"Mean Unemployment Rate (%)","US Debt (USD)", "US GDP (USD)", 
    "Post-productivity (trillion $)", "Current Subsidy Quantitative Policy Level",
    "Total Subsidies (USD)", "US Tax Wedge ('%' of GDP)", "US Federal Deficit (USD)", "US Federal Interest Payment (USD)", "Federal Reserve Fund Rate (%)", "US Treasury Yield Long Term (%)",
    "US Government Revenue (USD)", "US Health Index", "Defense Spending (USD)", "Income Security Spending (USD)",
    "Social Security Spending (USD)", "Medicare Medicaid Spending (USD)", "Federal Reserve Balance Sheet (USD)", "Inflation", "US Treasury Yield",
    "Mean Stringency Level", "Reward", "Reward Social Welfare"
]

headers_day = [
    "Day", "Susceptibles", "Infected", "Recovered", "Vaccinated (% of population)", "Deaths (thousands)" ,"Mean Unemployment Rate (%)","US Debt (USD)", "US GDP (USD)", 
    "Post-productivity (trillion $)", "Current Subsidy Quantitative Policy Level",
    "Total Subsidies (USD)", "US Tax Wedge ('%' of GDP)", "US Federal Deficit (USD)", "US Federal Interest Payment (USD)",
    "US Government Revenue (USD)", "Defense Spending (USD)", "Income Security Spending (USD)",
    "Social Security Spending (USD)", "Medicare Medicaid Spending (USD)", "Federal Reserve Balance Sheet (USD)", "Federal Reserve Fund Rate (%)", "Inflation", "US Treasury Yield Long Term (%)",
    "Mean Stringency Level", "Reward", "Reward Social Welfare"
]

try:
    num_gpus_available = len(GPUtil.getAvailable())
    print(f"Inside covid19_env.py: {num_gpus_available} GPUs are available.")
    if num_gpus_available == 0:
        print("No GPUs found! Running the simulation on a CPU.")
    else:
        from warp_drive.utils.constants import Constants
        from warp_drive.utils.data_feed import DataFeed

        _OBSERVATIONS = Constants.OBSERVATIONS
        _ACTIONS = Constants.ACTIONS
        _REWARDS = Constants.REWARDS
except ModuleNotFoundError:
    print(
        "Warning: The 'WarpDrive' package is not found and cannot be used! "
        "If you wish to use WarpDrive, please run "
        "'pip install rl-warp-drive' first."
    )
except ValueError:
    print("No GPUs found! Running the simulation on a CPU.")


@scenario_registry.add
class CovidAndEconomyEnvironment(BaseEnvironment):
    """
    A simulation to model health and economy dynamics amidst the COVID-19 pandemic.
    The environment comprising 51 agents (each agent corresponding to a US state and
    Washington D.C.) and the Federal Government (planner). The state agents decide the
    stringency level of the policy response to the pandemic, while the federal
    government provides subsidies to eligible individuals.

    This simulation makes modeling assumptions. For details, see the technical paper:
    https://arxiv.org/abs/2108.02904

    Args:
        use_real_world_data (bool): Replay what happened in the real world.
            Real-world data comprises SIR (susceptible/infected/recovered),
            unemployment, government policy, and vaccination numbers.
            This setting also sets use_real_world_policies=True.
        use_real_world_policies (bool): Run the environment with real-world policies
            (stringency levels and subsidies). With this setting and
            use_real_world_data=False, SIR and economy dynamics are still
            driven by fitted models.
        path_to_data_and_fitted_params (dirpath): Full path to the directory containing
            the data, fitted parameters and model constants. This defaults to
            "ai_economist/datasets/covid19_datasets/data_and_fitted_params".
            For details on obtaining these parameters, please see the notebook
            "ai-economist-foundation/ai_economist/datasets/covid19_datasets/
            gather_real_world_data_and_fit_parameters.ipynb".
        start_date (string): Date (YYYY-MM-DD) to start the simulation.
        pop_between_age_18_65 (float): Fraction of the population between ages 18-65.
            This is the subset of the population whose employment/unemployment affects
            economic productivity.
            Range: 0 <= pop_between_age_18_65 <= 1.
        infection_too_sick_to_work_rate (float): Fraction of people infected with
            COVID-19. Infected people don't work.
            Range: 0 <= infection_too_sick_to_work_rate <= 1
        risk_free_interest_rate (float): Percentage of interest paid by the federal
            government to borrow money from the federal reserve for COVID-19 relief
            (direct payments). Higher interest rates mean that direct payments
            have a larger cost on the federal government's economic index.
            Range: 0 <= risk_free_interest_rate
        economic_reward_crra_eta (float): CRRA eta parameter for modeling the economic
            reward non-linearity.
            A useful reference: https://en.wikipedia.org/wiki/Isoelastic_utility
            Range: 0 <= economic_reward_crra_eta
        health_priority_scaling_agents (float): A factor indicating how much more the
            states prioritize health (roughly speaking, loss of lives due to
            opening up more) over the economy (roughly speaking, a loss in GDP
            due to shutting down resulting in more unemployment) compared to the
            real-world.
            For example, a value of 1 corresponds to the real-world, while
            a value of 2 means that states cared twice as much about public health
            (preventing deaths), while a value of 0.5 means that states cared twice
            as much about the economy (preventing GDP drops).
            Range: 0 <= health_priority_scaling_agents
        health_priority_scaling_planner (float): same as above,
            but for the federal government.
            Range: 0 <= health_priority_scaling_planner
    """

    def __init__(
        self,
        *base_env_args,
        use_real_world_data=False,
        use_real_world_policies=False,
        path_to_data_and_fitted_params="",
        start_date="2020-03-22",
        pop_between_age_18_65=0.6,
        infection_too_sick_to_work_rate=0.1,
        risk_free_interest_rate=0.5/100,
        fed_fund_rate_01_2020=0.25,
        inflation_cpi_2019=0.017,
        economic_reward_crra_eta=2,
        health_priority_scaling_agents=1,
        health_priority_scaling_planner=1,
        reward_normalization_factor=1, 
        us_government_spending_economic_multiplier=1, # Fiscal multiplier - Average of 1.7 and 3.3 via tax cuts and spending increases - One paper stated that US Government spending only have 0.6 - 0.8 multiplier Fiscal multiplier - https://www.nber.org/system/files/working_papers/w15464/w15464.pdf
        us_government_mandatory_and_discretionary_spending = 4.4 * 10**12 / 365, 
        us_government_defense_spending = 676 * 10**9 / 365, # https://www.cbo.gov/publication/56324 - 2019's number
        us_government_social_security_spending = 1.038 * 10**12 / 365, # https://www.cbo.gov/publication/56324 - 2019's number
        us_government_medicare_medicaid_spending = 1.258 * 10**12 / 365, # https://www.cbo.gov/publication/56324 - 2019's number
        us_government_income_security = 3.03 * 10**11 / 365, # https://www.cbo.gov/publication/56324 - 2019's number
        us_government_non_defense_others_spending = 6.61 * 10**11 / 365, # https://www.cbo.gov/publication/56324 - 2019's number
        us_federal_net_interest = 0.375 * 10**12 / 365, # https://www.cbo.gov/publication/56324 - 2019's number
        us_government_debt = 16.898 * 10**12, # https://www.cbo.gov/publication/56309
        us_treasury_yield_long_term= 1.92 / 100, # assume that US Treasury only issues long-term treasury bonds
        us_federal_revenue= 3.5 * 10**12 / 365, # https://www.cbo.gov/publication/56324 - 2019's number
        us_M2_money_supply= 3955.3*10**9,
        fed_reserve_balance_sheet= 4.173626 * 10**12, # assume that FED only buy long-term treasury bonds
        cbo_output_gap_2019=0.9165, # https://www.cbo.gov/data/budget-economic-data - 10-Year Economic Projections - Jul 2020
        social_security_participants= (69.1 + 5.7) * 10**6, # https://www.ssa.gov/policy/docs/chartbooks/fast_facts/2020/fast_facts20.html#pagei
        medicare_medicaid_participants=71395465, # https://www.cms.gov/newsroom/fact-sheets/medicaid-facts-and-figures#:~:text=Medicaid%20Facts%20and%20Figures%201%2071%2C395%2C465%20individuals%20were,was%2015%2C181%2C880%20for%20the%203rd%20quarter%20of%202018.%5B2%5D
        social_security_trust_funds_OASI_DI_trust_funds= 2.8974*10**12, # https://www.ssa.gov/oact/TRSUM/2020/index.html
        current_social_security_expanse = 1.078 * 10**12,
        social_security_poverty_reduction = 28000000,
        medicare_medicaid_poverty_reduction = 20000000,
        income_security_poverty_reduction = 9000000,      
        ideal_inflation=0.01,  
        social_security_beneficiaries= 64 * 10**6, # https://www.ssa.gov/cgi-bin/currentpay.cgi
        social_security_beneficiaries_growth = 10**6, # https://www.ssa.gov/cgi-bin/currentpay.cgi
        medicare_medicaid_beneficiaries_growth = 4 * 10**6, # https://www.statista.com/statistics/245347/total-medicaid-enrollment-since-1966/
        social_security_benefits_avg = 1384.19, # USD, https://www.ssa.gov/cgi-bin/currentpay.cgi
        income_security_benefits_avg = 40000, # USD, https://www.cbo.gov/publication/56325
        medicare_medicaid_security_benefits_avg = 4600 + 12302, # USD, https://www.cbo.gov/publication/56325, https://aspe.hhs.gov/reports/medicare-enrollment#:~:text=Medicare%20served%20nearly%2063%20million,percent%20had%20no%20drug%20coverage.
        income_security_participants=7.5 * 10**6, #https://www.cbpp.org/research/social-security/policy-basics-introduction-to-supplemental-security-income
        us_imperialism_level=2, # max is 5 - strong - Heritage Foundation
        max_us_imperialism_level=5,
        max_us_imperialism_level_spending_required=1.2 * 10**12,
        csv_validation=False,
        interest_hikes_shock_gdp=0.5, # 0.5% of GDP for every 100 basic point rate hikes
        state_governments_policies_only=False, # let the real-world state government handle the covid-19 restriction, and the federal government still operated by AI
        csv_file_path = 'simulation_results.csv',
        average_GDP_growth = 2/100, # average GDP Growth of the USA in the past 20 years
        **base_env_kwargs,
    ):
        # verify_activation_code()

        # Used for datatype checks
        self.np_float_dtype = np.float32
        self.np_int_dtype = np.int32
        self.csv_file_path = csv_file_path
        self.csv_file_path_day = csv_file_path.replace('.csv', '') + "_day.csv"

        # Flag to use real-world data or the fitted models instead
        self.use_real_world_data = use_real_world_data
        # Flag to use real-world policies (actions) or the supplied actions instead
        self.use_real_world_policies = use_real_world_policies

        self.state_governments_policies_only = state_governments_policies_only
        # If we use real-world data, we also want to use the real-world policies
        if self.use_real_world_data:
            print(
                "Using real-world data to initialize as well as to "
                "step through the env."
            )
            # Note: under this setting, the real_world policies are also used.
            assert self.use_real_world_policies, (
                "Since the env. config. 'use_real_world_data' is True, please also "
                "set 'use_real_world_policies' to True."
            )
        else:
            print(
                "Using the real-world data to only initialize the env, "
                "and using the fitted models to step through the env."
            )

        # Load real-world date
        print(path_to_data_and_fitted_params)
        if path_to_data_and_fitted_params == "":
            current_dir = os.path.dirname(__file__)
            self.path_to_data_and_fitted_params = os.path.join(
                current_dir, "../../../datasets/covid19_datasets/data_and_fitted_params"
            )
        else:
            current_dir = os.path.dirname(__file__)
            self.path_to_data_and_fitted_params = os.path.join(
                current_dir, path_to_data_and_fitted_params
            ) 

        print(
            "Loading real-world data from {}".format(
                self.path_to_data_and_fitted_params
            )
        )
        print(self.path_to_data_and_fitted_params)
        real_world_data_npz = np.load(
            os.path.join(self.path_to_data_and_fitted_params, "real_world_data.npz")
        )
        self._real_world_data = {}
        for key in list(real_world_data_npz): 
            print("Key: ", key, " - size: ", len(real_world_data_npz[key]))
            self._real_world_data[key] = real_world_data_npz[key] 
        # Load fitted parameters
        print(
            "Loading fit parameters from {}".format(self.path_to_data_and_fitted_params)
        ) 
        self.load_model_constants(self.path_to_data_and_fitted_params)
        self.load_fitted_params(self.path_to_data_and_fitted_params)  
        self.csv_validation = csv_validation
        try:
            self.start_date = datetime.strptime(start_date, self.date_format)
        except ValueError:
            print(f"Incorrect data format, should be {self.date_format}")

        # Start date should be beyond the date for which data is available
        assert self.start_date >= self.policy_start_date

        # Compute a start date index based on policy start date
        self.start_date_index = (self.start_date - self.policy_start_date).days
        assert 0 <= self.start_date_index < len(self._real_world_data["policy"])

        # For date logging (This will be overwritten in additional_reset_steps;
        # see below)
        self.current_date = None
        self.delete_csv_file = True
        self.delete_csv_day_file = True

        # real-life nominal GDP
        # https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=US
        # CBO: The Federal Budget in Fiscal Year 2020: An Infographic
        # CBO: The Federal Budget in Fiscal Year 2021: An Infographic
        # CBO: The Federal Budget in Fiscal Year 2022: An Infographic
        self.gdp_2020 = 21.06 * 10**12
        self.gdp_2021 = 23.32 * 10**12
        self.gdp_2022 = 25.44 * 10**12

        self.revenue_2020 = 3.4 * 10**12
        self.revenue_2021 = 4 * 10**12
        self.revenue_2022 = 4.9 * 10**12 
 
        self.spending_2020 = 6.55 * 10**12
        self.spending_2021 =  6.8 * 10**12
        self.spending_2022 = 6.3 * 10**12  

        self.defense_spending_2020 = 0.714 * 10**12
        self.defense_spending_2021 =  0.742 * 10**12
        self.defense_spending_2022 = 0.751 * 10**12  

        self.income_security_spending_2020 = 1.052 * 10**12
        self.income_security_spending_2021 =  1.376 * 10**12
        self.income_security_spending_2022 = 6.3 * 10**12  

        self.social_security_spending_2020 = 1.1 * 10**12
        self.social_security_spending_2021 =  1.129 * 10**12
        # I added the student loan program for its significant budget cost
        self.social_security_spending_2022 = 0.581 * 10**12 + 0.482 * 10**12
        
        self.medicare_medicade_spending_2020 = (0.769 + 0.458) * 10**12
        self.medicare_medicade_spending_2021 = (0.689 + 0.521) * 10**12
        self.medicare_medicade_spending_2022 = (0.747 + 0.592) * 10**12  
        # When using real-world policy, limit the episode length
        # to the length of the available policy.
        
        if self.use_real_world_data is False:
            self.gdp_per_capita = self.gdp_per_capita - self.defense_spending_2020 / self.us_population - self.medicare_medicade_spending_2020 / self.us_population - self.income_security_spending_2020 / self.us_population
        
        if self.use_real_world_policies:
            real_world_policy_length = (
                len(self._real_world_data["policy"]) - self.start_date_index
            )
            print("Using real-world policies, ignoring external action inputs.")
            assert base_env_kwargs["episode_length"] <= real_world_policy_length, (
                f"The real-world policies are only available for "
                f"{real_world_policy_length} timesteps; so the 'episode_length' "
                f"in the environment configuration can only be at most "
                f"{real_world_policy_length}"
            )
        else:
            print("Using external action inputs.")

        # US states and populations
        self.num_us_states = len(self.us_state_population)
        self.cbo_output_gap_2019 = cbo_output_gap_2019 
        print(f"Number of US states: {self.num_us_states}")
        # assert (
        #     base_env_kwargs["n_agents"] == self.num_us_states
        # ), "n_agents should be set to the number of US states, i.e., {}.".format(
        #     self.num_us_states
        # )
        # Note: For a faster environment step time, we collate all the individual agents
        # into a single agent index "a" and we flatten the component action masks too.
        # assert base_env_kwargs[
        #     "collate_agent_step_and_reset_data"
        # ], "The env. config 'collate_agent_step_and_reset_data' should be set to True."
        super().__init__(*base_env_args, **base_env_kwargs)

        # Add attributes to self.world for use in components
        self.world.us_state_population = self.us_state_population
        self.world.GDP_Growth = 0
        self.world.us_population = self.us_population
        self.world.start_date_index = self.start_date_index
        self.world.start_date = self.start_date
        self.world.n_stringency_levels = self.num_stringency_levels
        self.world.use_real_world_policies = self.use_real_world_policies
        self.world.state_governments_policies_only = self.state_governments_policies_only
        self.medicare_medicaid_security_benefits_avg = us_government_medicare_medicaid_spending/medicare_medicaid_participants
        self.medicare_medicaid_participants = medicare_medicaid_participants
        self.income_security_participants = income_security_participants
        self.income_security_benefits_avg = income_security_benefits_avg
        self.medicare_medicaid_benefits_avg = medicare_medicaid_security_benefits_avg
        if self.state_governments_policies_only or self.use_real_world_policies:
            # Agent open/close stringency levels
            self.world.real_world_stringency_policy = self._real_world_data["policy"][
                self.start_date_index :
            ]

        if self.use_real_world_policies:
            # Planner subsidy/quantitative levels
            self.world.real_world_subsidy = self._real_world_data["subsidy"][
                self.start_date_index :
            ]
            self.world.real_world_quantitative = self._real_world_data["quantitative"][
                self.start_date_index :
            ] 

            self.world.real_world_inflation = self._real_world_data["inflation"][
                self.start_date_index :
            ]

            self.world.real_world_fed_fund_rate = self._real_world_data["fed_fund_rate"][
                self.start_date_index :
            ]

            self.world.real_world_us_treasury_yield_long_10_years = self._real_world_data["us_treasury_yield_long_10_years"][
                self.start_date_index :
            ]

            self.world.real_world_revenue = self._real_world_data["revenue"][
                self.start_date_index :
            ]

            self.world.real_world_spending = self._real_world_data["spending"][
                self.start_date_index :
            ]


            self.world.real_world_debt = self._real_world_data["debt"][
                self.start_date_index :
            ]

        # Policy --> Unemployment
        #   For accurately modeling the state-wise unemployment, we convolve
        #   the current stringency policy with a family of exponential filters
        #   with separate means (lambdas).
        # This code sets up things we will use in `unemployment_step()`,
        #   which includes a detailed breakdown of how the unemployment model is
        #   implemented.
        self.stringency_level_history = None
        # Each filter captures a temporally extended response to a stringency change.
        self.num_filters = len(self.conv_lambdas)
        self.f_ts = np.tile(
            np.flip(np.arange(self.filter_len), (0,))[None, None],
            (1, self.num_filters, 1),
        ).astype(self.np_float_dtype)
        self.unemp_conv_filters = np.exp(-self.f_ts / self.conv_lambdas[None, :, None])
        # Each state weights these filters differently.
        self.repeated_conv_weights = np.repeat(
            self.grouped_convolutional_filter_weights.reshape(
                self.num_us_states, self.num_filters
            )[:, :, np.newaxis],
            self.filter_len,
            axis=-1,
        )

        # For manually modulating SIR/Unemployment parameters
        self._beta_intercepts_modulation = 1
        self._beta_slopes_modulation = 1
        self._unemployment_modulation = 1

        # Economy-related
        # Interest rate for borrowing money from the federal reserve
        self.risk_free_interest_rate = self.np_float_dtype(risk_free_interest_rate)
        self.inflation_cpi_2019 = self.np_float_dtype(inflation_cpi_2019)
        self.fed_fund_rates = self.np_float_dtype(fed_fund_rate_01_2020)
        self.interest_hikes_shock_gdp = self.np_float_dtype(interest_hikes_shock_gdp)
        self.us_treasury_yield_long_term = self.np_float_dtype(us_treasury_yield_long_term)
        self.us_government_debt = self.np_float_dtype(us_government_debt)
        self.fed_reserve_balance_sheet = self.np_float_dtype(fed_reserve_balance_sheet)
        self.us_M2_money_supply = self.np_float_dtype(us_M2_money_supply)
        self.us_imperialism_level = us_imperialism_level
        self.max_us_imperialism_level = max_us_imperialism_level
        self.max_us_imperialism_level_spending_required = max_us_imperialism_level_spending_required
        self.ideal_inflation = ideal_inflation
        # Compute each worker's daily productivity when at work (to match 2019 GDP)
        # We assume the open/close stringency policy level was always at it's lowest
        # value (i.e., 1) before the pandemic started.
        num_unemployed_at_stringency_level_1 = self.unemployment_step(
            np.ones(self.num_us_states)
        )
        self.workforce = (
            self.us_population * pop_between_age_18_65
            - np.sum(num_unemployed_at_stringency_level_1)
        ).astype(self.np_int_dtype)
        self.workers_per_capita = (self.workforce / self.us_population).astype(
            self.np_float_dtype
        )
        self.gdp_per_worker = (self.gdp_per_capita / self.workers_per_capita).astype(
            self.np_float_dtype
        )
        # The tax wedge for the average single worker in the United States increased by 1.2 percentage points from 27.2% in 2020 to 28.4% in
        # 2021, according to the Taxing Wages - the United States. https://www.oecd.org/unitedstates/taxing-wages-united-states.pdf


        # The US government spending multiplier is 0.8, according to the Congressional Budget Office. https://www.crfb.org/papers/comparing-fiscal-multipliers
        self.us_government_spending_economic_multiplier = self.np_float_dtype(us_government_spending_economic_multiplier)
        self.average_GDP_growth = average_GDP_growth
        self.num_days_in_an_year = 365
        self.daily_production_per_worker = (
            self.gdp_per_worker / self.num_days_in_an_year
        ).astype(self.np_float_dtype)

        self.infection_too_sick_to_work_rate = self.np_float_dtype(
            infection_too_sick_to_work_rate
        )
        assert 0 <= self.infection_too_sick_to_work_rate <= 1

        self.pop_between_age_18_65 = self.np_float_dtype(pop_between_age_18_65)
        assert 0 <= self.pop_between_age_18_65 <= 1

        # according to the Congressional Budget Office. https://www.cbo.gov/publication/56324
        self.us_government_revenue = us_federal_revenue
        self.us_government_mandatory_and_discretionary_spending = \
            us_government_mandatory_and_discretionary_spending
        self.trillion = 10**12
        self.us_government_defense_spending = us_government_defense_spending
        self.us_government_social_security_spending = us_government_social_security_spending
        self.us_government_medicare_medicaid_spending = us_government_medicare_medicaid_spending
        self.us_government_non_defense_others_spending = us_government_non_defense_others_spending
        self.us_government_income_security = us_government_income_security
        self.us_federal_net_interest = us_federal_net_interest
        self.us_federal_deficit = self.us_government_mandatory_and_discretionary_spending \
            - self.us_government_revenue
        self.social_security_poverty_reduction = social_security_poverty_reduction
        self.medicare_medicaid_poverty_reduction = medicare_medicaid_poverty_reduction
        self.income_security_poverty_reduction = income_security_poverty_reduction
        self.social_security_beneficiaries = social_security_beneficiaries
        self.social_security_beneficiaries_growth = social_security_beneficiaries_growth
        self.medicare_medicaid_beneficiaries_growth = medicare_medicaid_beneficiaries_growth
        self.social_security_benefits_avg = social_security_benefits_avg
        self.dictionary_fiscal_theory = []

        # Compute max possible productivity values (used for agent reward normalization)
        max_productivity_t = self.economy_step(
            self.us_state_population,
            np.zeros((self.num_us_states), dtype=self.np_int_dtype),
            np.zeros((self.num_us_states), dtype=self.np_int_dtype),
            num_unemployed_at_stringency_level_1,
            infection_too_sick_to_work_rate=self.infection_too_sick_to_work_rate,
            population_between_age_18_65=self.pop_between_age_18_65,
        )

        self.maximum_productivity_t = max_productivity_t
        self.us_gdp_2019 = self.us_population * self.gdp_per_capita
        print("self.us_gdp_2019: ", self.us_gdp_2019)
        self.us_tax_wedge = self.np_float_dtype(self.us_government_revenue * 365 / self.us_gdp_2019) 
        # 2019 US government spending was $4.4 trillion,
        
        # Economic reward non-linearity
        self.economic_reward_crra_eta = self.np_float_dtype(economic_reward_crra_eta)
        assert 0.0 <= self.economic_reward_crra_eta < 20.0

        # Health indices are normalized by maximum annual GDP
        self.agents_health_norm = self.maximum_productivity_t * self.num_days_in_an_year
        self.planner_health_norm = np.sum(self.agents_health_norm)

        # Economic indices are normalized by maximum annual GDP
        self.agents_economic_norm = (
            self.maximum_productivity_t * self.num_days_in_an_year
        )
        self.planner_economic_norm = np.sum(self.agents_economic_norm)

        def scale_health_over_economic_index(health_priority_scaling, alphas):
            """
            Given starting alpha(s), compute new alphas so that the
            resulting alpha:1-alpha ratio is scaled by health_weightage
            """
            z = alphas / (1 - alphas)  # alphas = z / (1 + z)
            scaled_z = health_priority_scaling * z
            new_alphas = scaled_z / (1 + scaled_z)
            return new_alphas

        # Agents' health and economic index weightages
        # fmt: off
        self.weightage_on_marginal_agent_health_index = \
            scale_health_over_economic_index(
                health_priority_scaling_agents,
                self.inferred_weightage_on_agent_health_index,
            )
        # fmt: on
        assert (
            (self.weightage_on_marginal_agent_health_index >= 0)
            & (self.weightage_on_marginal_agent_health_index <= 1)
        ).all()
        self.weightage_on_marginal_agent_economic_index = (
            1 - self.weightage_on_marginal_agent_health_index
        )

        # Planner's health and economic index weightages
        # fmt: off
        self.weightage_on_marginal_planner_health_index = \
            scale_health_over_economic_index(
                health_priority_scaling_planner,
                self.inferred_weightage_on_planner_health_index,
            )
        # fmt: on
        assert 0 <= self.weightage_on_marginal_planner_health_index <= 1
        self.weightage_on_marginal_planner_economic_index = (
            1 - self.weightage_on_marginal_planner_health_index
        )

        # Normalization factor for the reward (often useful for RL training)
        self.reward_normalization_factor = reward_normalization_factor

        # CUDA-related attributes (for GPU simulations)
        # Note: these will be set / overwritten via the env_wrapper
        # use_cuda will be set to True (by the env_wrapper), if needed
        # to be simulated on the GPU
        self.use_cuda = False
        self.cuda_data_manager = None
        self.cuda_function_manager = None
        self.cuda_step = lambda *args, **kwargs: None
        self.cuda_compute_reward = lambda *args, **kwargs: None

        # Adding use_cuda to self.world for use in components
        self.world.use_cuda = self.use_cuda
        self.world.cuda_data_manager = self.cuda_data_manager
        self.world.cuda_function_manager = self.cuda_function_manager

    name = "CovidAndEconomySimulation"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    required_entities = []

    def reset_starting_layout(self):
        pass

    def reset_agent_states(self):
        self.world.clear_agent_locs()

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the GPU (device).
        """
        data_dict = DataFeed()
        # Global States
        data_dict.add_data(
            name="susceptible",
            data=self.world.global_state["Susceptible"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="infected",
            data=self.world.global_state["Infected"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="recovered",
            data=self.world.global_state["Recovered"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="deaths",
            data=self.world.global_state["Deaths"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="unemployed",
            data=self.world.global_state["Unemployed"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="vaccinated",
            data=self.world.global_state["Vaccinated"],
            save_copy_and_apply_at_reset=True,
        )
        # Actions
        data_dict.add_data(
            name="stringency_level",
            data=self.world.global_state["Stringency Level"].astype(self.np_int_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="us_government_spending_economic_multiplier",
            data=self.us_government_spending_economic_multiplier,
        )
        data_dict.add_data(
            name="USTaxWedge",
            data=self.world.global_state["US Tax Wedge"],
            save_copy_and_apply_at_reset=True,
        )
        # Economy-related
        data_dict.add_data(
            name="subsidy",
            data=self.world.global_state["Subsidy"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="output_gap",
            data=self.world.global_state["Output Gap"],
            save_copy_and_apply_at_reset=True,
        )
        # Federal Reserve
        data_dict.add_data(
            name="quantitative",
            data=self.world.global_state["Quantitative"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="USGovernmentRevenue",
            data=self.world.global_state["US Government Revenue"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="USGovernmentMandatoryAndDiscretionarySpending",
            data=self.world.global_state["US Government Mandatory and Discretionary Spending"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="USTreasuryYieldLongTerm",
            data=self.world.global_state["US Treasury Yield Long Term"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="USFederalDeficit",
            data=self.world.global_state["US Federal Deficit"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="USDebt",
            data=self.world.global_state["US Debt"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="FederalReserveBalanceSheet",
            data=self.world.global_state["Federal Reserve Balance Sheet"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="FederalReserveFundRate",
            data=self.world.global_state["Federal Reserve Fund Rate"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="USGovernmentDefenseSpending",
            data=self.world.global_state["US Government Defense Spending"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="USGovernmentSocialSecuritySpending",
            data=self.world.global_state["US Government Social Security Spending"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="USGovernmentMedicareMedicaidSpending",
            data=self.world.global_state["US Government Medicare Medicaid Spending"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="USFederalInterestPayment",
            data=self.world.global_state["US Federal Interest Payment"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="Inflation",
            data=self.world.global_state["Inflation"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="ReducedGDPMultiplier",
            data=self.world.global_state["Reduced GDP Multiplier"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="US GDP",
            data=self.world.global_state["US GDP"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="us_treasury_yield_long_term",
            data=self.us_treasury_yield_long_term,
        )
        data_dict.add_data(
            name="postsubsidy_productivity",
            data=self.world.global_state["Postsubsidy Productivity"],
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="productivity",
            data=np.zeros_like(
                self.world.global_state["Susceptible"], dtype=self.np_float_dtype
            ),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="incapacitated",
            data=np.zeros(self.num_us_states, dtype=self.np_float_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="cant_work",
            data=np.zeros(self.num_us_states, dtype=self.np_float_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="num_people_that_can_work",
            data=np.zeros(self.num_us_states, dtype=self.np_float_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="us_gdp_2019",
            data=self.us_gdp_2019,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="gdp_per_capita",
            data=self.gdp_per_capita,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="us_state_population",
            data=self.us_state_population,
        )
        data_dict.add_data(
            name="infection_too_sick_to_work_rate",
            data=self.infection_too_sick_to_work_rate,
        )
        data_dict.add_data(
            name="population_between_age_18_65",
            data=self.pop_between_age_18_65,
        )
        data_dict.add_data(
            name="daily_production_per_worker",
            data=self.daily_production_per_worker,
        )
        data_dict.add_data(
            name="maximum_productivity",
            data=self.maximum_productivity_t,
        )
        # SIR-related
        data_dict.add_data(
            name="real_world_stringency_policy_history",
            data=(
                self._real_world_data["policy"][
                    self.start_date_index - self.beta_delay + 1 : self.start_date_index,
                    :,
                ]
            ).astype(self.np_int_dtype),
        )
        data_dict.add_data(
            name="beta_delay",
            data=self.beta_delay,
        )
        data_dict.add_data(
            name="beta_slopes",
            data=self.beta_slopes,
        )
        data_dict.add_data(
            name="beta_intercepts",
            data=self.beta_intercepts,
        )
        data_dict.add_data(
            name="beta",
            data=np.zeros((self.num_us_states), dtype=self.np_float_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="gamma",
            data=self.gamma,
        )
        data_dict.add_data(
            name="death_rate",
            data=self.death_rate,
        )
        # Unemployment fit parameters
        data_dict.add_data(
            name="filter_len",
            data=self.filter_len,
        )
        data_dict.add_data(
            name="num_filters",
            data=self.num_filters,
        )
        data_dict.add_data(
            name="delta_stringency_level",
            data=(
                self.stringency_level_history[1:] - self.stringency_level_history[:-1]
            ).astype(self.np_int_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="grouped_convolutional_filter_weights",
            data=self.grouped_convolutional_filter_weights,
        )
        data_dict.add_data(
            name="unemp_conv_filters",
            data=self.unemp_conv_filters,
        )
        data_dict.add_data(
            name="unemployment_bias",
            data=self.unemployment_bias,
        )
        data_dict.add_data(
            name="signal",
            data=np.zeros(
                (self.n_agents, self.num_filters, self.filter_len),
                dtype=self.np_float_dtype,
            ),
            save_copy_and_apply_at_reset=True,
        )
        # Reward-related
        data_dict.add_data(
            name="min_marginal_agent_health_index",
            data=self.min_marginal_agent_health_index,
        )
        data_dict.add_data(
            name="max_marginal_agent_health_index",
            data=self.max_marginal_agent_health_index,
        )
        data_dict.add_data(
            name="min_marginal_agent_economic_index",
            data=self.min_marginal_agent_economic_index,
        )
        data_dict.add_data(
            name="max_marginal_agent_economic_index",
            data=self.max_marginal_agent_economic_index,
        )
        data_dict.add_data(
            name="min_marginal_planner_health_index",
            data=self.min_marginal_planner_health_index,
        )
        data_dict.add_data(
            name="max_marginal_planner_health_index",
            data=self.max_marginal_planner_health_index,
        )
        data_dict.add_data(
            name="min_marginal_planner_economic_index",
            data=self.min_marginal_planner_economic_index,
        )
        data_dict.add_data(
            name="max_marginal_planner_economic_index",
            data=self.max_marginal_planner_economic_index,
        )
        data_dict.add_data(
            name="weightage_on_marginal_agent_health_index",
            data=self.weightage_on_marginal_agent_health_index,
        )
        data_dict.add_data(
            name="weightage_on_marginal_agent_economic_index",
            data=self.weightage_on_marginal_agent_economic_index,
        )
        data_dict.add_data(
            name="weightage_on_marginal_planner_health_index",
            data=self.weightage_on_marginal_planner_health_index,
        )
        data_dict.add_data(
            name="weightage_on_marginal_planner_economic_index",
            data=self.weightage_on_marginal_planner_economic_index,
        )
        data_dict.add_data(
            name="value_of_life",
            data=self.value_of_life,
        )
        data_dict.add_data(
            name="economic_reward_crra_eta",
            data=self.economic_reward_crra_eta,
        )
        data_dict.add_data(
            name="num_days_in_an_year",
            data=self.num_days_in_an_year,
        )
        data_dict.add_data(
            name="risk_free_interest_rate",
            data=self.risk_free_interest_rate,
        )
        data_dict.add_data(
            name="inflation_cpi_2019",
            data=self.inflation_cpi_2019,
        )
        data_dict.add_data(
            name="agents_health_norm",
            data=self.agents_health_norm,
        )
        data_dict.add_data(
            name="agents_economic_norm",
            data=self.agents_economic_norm,
        )
        data_dict.add_data(
            name="planner_health_norm",
            data=self.planner_health_norm,
        )
        data_dict.add_data(
            name="planner_economic_norm",
            data=self.planner_economic_norm,
        )

        return data_dict

    def get_tensor_dictionary(self):
        """
        Create a dictionary of (Pytorch-accessible) data to push to the GPU (device).
        """
        tensor_dict = DataFeed()
        return tensor_dict

    def scenario_step(self):
        """
        Update the state of the USA based on the Covid-19 and Economy dynamics.
        This internally implements three steps
        - sir_step() - updates the susceptible, infected, recovered, deaths
        and vaccination numbers based on the SIR equations
        - unemployment_step() - uses the unemployment model to updates the unemployment
         based on the stringency levels
        - economy_step - computes the current productivity numbers for the agents
        """
        if self.use_cuda:
            self.cuda_step(
                self.cuda_data_manager.device_data("susceptible"),
                self.cuda_data_manager.device_data("infected"),
                self.cuda_data_manager.device_data("recovered"),
                self.cuda_data_manager.device_data("deaths"),
                self.cuda_data_manager.device_data("vaccinated"),
                self.cuda_data_manager.device_data("unemployed"),
                self.cuda_data_manager.device_data("subsidy"),
                self.cuda_data_manager.device_data("productivity"),
                self.cuda_data_manager.device_data("stringency_level"),
                self.cuda_data_manager.device_data("num_stringency_levels"),
                self.cuda_data_manager.device_data("postsubsidy_productivity"),
                self.cuda_data_manager.device_data("num_vaccines_available_t"),
                self.cuda_data_manager.device_data(
                    "real_world_stringency_policy_history"
                ),
                self.cuda_data_manager.device_data("beta_delay"),
                self.cuda_data_manager.device_data("beta_slopes"),
                self.cuda_data_manager.device_data("beta_intercepts"),
                self.cuda_data_manager.device_data("beta"),
                self.cuda_data_manager.device_data("gamma"),
                self.cuda_data_manager.device_data("death_rate"),
                self.cuda_data_manager.device_data("incapacitated"),
                self.cuda_data_manager.device_data("cant_work"),
                self.cuda_data_manager.device_data("num_people_that_can_work"),
                self.cuda_data_manager.device_data("us_state_population"),
                self.cuda_data_manager.device_data("infection_too_sick_to_work_rate"),
                self.cuda_data_manager.device_data("population_between_age_18_65"),
                self.cuda_data_manager.device_data("filter_len"),
                self.cuda_data_manager.device_data("num_filters"),
                self.cuda_data_manager.device_data("delta_stringency_level"),
                self.cuda_data_manager.device_data(
                    "grouped_convolutional_filter_weights"
                ),
                self.cuda_data_manager.device_data("unemp_conv_filters"),
                self.cuda_data_manager.device_data("unemployment_bias"),
                self.cuda_data_manager.device_data("signal"),
                self.cuda_data_manager.device_data("daily_production_per_worker"),
                self.cuda_data_manager.device_data("maximum_productivity"),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_world-agent_state"
                ),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_world-agent_postsubsidy_productivity"
                ),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_a_world-lagged_stringency_level"
                ),
                self.cuda_data_manager.device_data(f"{_OBSERVATIONS}_a_time"),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_world-agent_state"
                ),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_world-agent_postsubsidy_productivity"
                ),
                self.cuda_data_manager.device_data(
                    f"{_OBSERVATIONS}_p_world-lagged_stringency_level"
                ),
                self.cuda_data_manager.device_data(f"{_OBSERVATIONS}_p_time"),
                self.cuda_data_manager.device_data("_timestep_"),
                self.cuda_data_manager.device_data("quantitative"),
                self.cuda_data_manager.device_data("ReducedGDPMultiplier"), 
                self.cuda_data_manager.device_data("USFederalDeficit"), 
                self.cuda_data_manager.device_data("USGovernmentRevenue"), 
                self.cuda_data_manager.device_data("USGovernmentMandatoryAndDiscretionarySpending"), 
                self.cuda_data_manager.device_data("USDebt"), 
                self.cuda_data_manager.device_data("Inflation"), 
                self.cuda_data_manager.device_data("FederalReserveBalanceSheet"), 
                self.cuda_data_manager.device_data("FederalReserveFundRate"), 
                self.cuda_data_manager.device_data("us_government_spending_economic_multiplier"), 
                self.cuda_data_manager.device_data("us_tax_wedge"), 
                self.cuda_data_manager.meta_info("n_agents"),
                self.cuda_data_manager.meta_info("episode_length"),
                block=self.world.cuda_function_manager.block,
                grid=self.world.cuda_function_manager.grid,
            )
        else:
            prev_t = self.world.timestep - 1
            curr_t = self.world.timestep

            self.current_date += timedelta(days=1)

            # SIR
            # ---
            if self.use_real_world_data or self.state_governments_policies_only:
                _S_t = np.maximum(
                    self._real_world_data["susceptible"][
                        curr_t + self.start_date_index
                    ],
                    0,
                )
                _I_t = np.maximum(
                    self._real_world_data["infected"][curr_t + self.start_date_index],
                    0,
                )
                _R_t = np.maximum(
                    self._real_world_data["recovered"][curr_t + self.start_date_index],
                    0,
                )
                _V_t = np.maximum(
                    self._real_world_data["vaccinated"][curr_t + self.start_date_index],
                    0,
                )
                _D_t = np.maximum(
                    self._real_world_data["deaths"][curr_t + self.start_date_index],
                    0,
                ) 

            else:  # Use simulation logic
                if curr_t - self.beta_delay < 0:
                    if self.start_date_index + curr_t - self.beta_delay < 0:
                        stringency_level_tmk = np.ones(self.num_us_states)
                    else:
                        stringency_level_tmk = self._real_world_data["policy"][
                            self.start_date_index + curr_t - self.beta_delay, :
                        ]
                else:
                    stringency_level_tmk = self.world.global_state["Stringency Level"][
                        curr_t - self.beta_delay
                    ]
                stringency_level_tmk = stringency_level_tmk.astype(self.np_int_dtype)

                _S_tm1 = self.world.global_state["Susceptible"][prev_t]
                _I_tm1 = self.world.global_state["Infected"][prev_t]
                _R_tm1 = self.world.global_state["Recovered"][prev_t]
                _V_tm1 = self.world.global_state["Vaccinated"][prev_t]

                # Vaccination
                # -----------
                num_vaccines_available_t = np.zeros(
                    self.n_agents, dtype=self.np_int_dtype
                )
                for aidx, agent in enumerate(self.world.agents):
                    # "Load" the vaccines in the inventory into this vector.
                    num_vaccines_available_t[aidx] = agent.state["Vaccines Available"]
                    # Agents always use whatever vaccines they can, so this becomes 0:
                    agent.state["Total Vaccinated"] += agent.state["Vaccines Available"]
                    agent.state["Vaccines Available"] = 0
                # SIR step
                # --------
                _dS, _dI, _dR, _dV = self.sir_step(
                    _S_tm1,
                    _I_tm1,
                    stringency_level_tmk,
                    num_vaccines_available_t,
                )
                _S_t = np.maximum(_S_tm1 + _dS, 0)
                _I_t = np.maximum(_I_tm1 + _dI, 0)
                _R_t = np.maximum(_R_tm1 + _dR, 0)
                _V_t = np.maximum(_V_tm1 + _dV, 0)

                num_recovered_but_not_vaccinated_t = _R_t - _V_t
                _D_t = self.death_rate * num_recovered_but_not_vaccinated_t

            # Update global state
            # -------------------
            self.world.global_state["Susceptible"][curr_t] = _S_t
            self.world.global_state["Infected"][curr_t] = _I_t
            self.world.global_state["Recovered"][curr_t] = _R_t
            self.world.global_state["Deaths"][curr_t] = _D_t
            self.world.global_state["Vaccinated"][curr_t] = _V_t

            # Unemployment
            # ------------
            if self.use_real_world_data or self.state_governments_policies_only:
                num_unemployed_t = self._real_world_data["unemployed"][
                    self.start_date_index + curr_t
                ]
            else:
                num_unemployed_t = self.unemployment_step(
                    current_stringency_level=self.world.global_state[
                        "Stringency Level"
                    ][curr_t]
                )

            self.world.global_state["Unemployed"][curr_t] = num_unemployed_t
            # Fiscal Multiplier : https://www.crfb.org/papers/comparing-fiscal-multipliers 
            # Productivity
            # ------------  
            productivity_t = self.economy_step(
                self.us_state_population,
                infected=_I_t,
                deaths=_D_t,
                unemployed=num_unemployed_t,
                infection_too_sick_to_work_rate=self.infection_too_sick_to_work_rate,
                population_between_age_18_65=self.pop_between_age_18_65,
            )
            if self.use_real_world_data:  
                if(self.world.timestep == 1 or self.world.timestep == 0): 
                    self.world.global_state["US Government Revenue"][self.world.timestep] = self.revenue_2020 / 365
                    self.world.global_state["US Federal Deficit"] = self.spending_2020 / 365
                    
                    
                    self.world.global_state["US Government Defense Spending"][self.world.timestep] \
                        = self.defense_spending_2020 / 365
                    
                    self.world.global_state["US Government Social Security Spending"][self.world.timestep] \
                        = self.social_security_spending_2020 / 365
                    
                    self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep] \
                        = self.medicare_medicade_spending_2020 / 365
                    
                    self.world.global_state["US Government Income Security"][self.world.timestep] \
                        = self.income_security_spending_2020 / 365

                if(self.world.timestep / 365 == 1 and self.world.timestep % 365 == 0):
                    self.world.global_state["US Government Revenue"][self.world.timestep] = self.revenue_2021 / 365
                    self.world.global_state["US Federal Deficit"] = self.spending_2021 / 365
                    self.world.global_state["US Tax Wedge"] = self.revenue_2021 / self.gdp_2021
                    self.world.global_state["US Government Defense Spending"][self.world.timestep] \
                        = self.defense_spending_2021 / 365
                    
                    self.world.global_state["US Government Social Security Spending"][self.world.timestep] \
                        = self.social_security_spending_2021 / 365
                    
                    self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep] \
                        = self.medicare_medicade_spending_2021 / 365
                    
                    self.world.global_state["US Government Income Security"][self.world.timestep] \
                        = self.income_security_spending_2021 / 365
                    
                if(self.world.timestep / 365 == 2 and self.world.timestep % 365 == 0):
                    self.world.global_state["US Government Revenue"][self.world.timestep] = self.revenue_2022 / 365
                    self.world.global_state["US Federal Deficit"] = self.spending_2022 / 365
                    
                    self.world.global_state["US Tax Wedge"] = self.revenue_2022 / self.gdp_2022
                    self.world.global_state["US Government Defense Spending"][self.world.timestep] \
                        = self.defense_spending_2022 / 365
                    
                    self.world.global_state["US Government Social Security Spending"][self.world.timestep] \
                        = self.social_security_spending_2022 / 365
                    
                    self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep] \
                        = self.medicare_medicade_spending_2022 / 365
                    
                    self.world.global_state["US Government Income Security"][self.world.timestep] \
                        = self.income_security_spending_2022 / 365
                # if(self.world.timestep >= 1 and self.world.timestep % 365 != 0):
                self.world.global_state["US Government Revenue"][self.world.timestep + 1] = \
                    self.world.global_state["US Government Revenue"][self.world.timestep]
                self.world.global_state["US Government Defense Spending"][self.world.timestep + 1] \
                    = self.world.global_state["US Government Defense Spending"][self.world.timestep] 
                
                self.world.global_state["US Government Social Security Spending"][self.world.timestep + 1] \
                    = self.world.global_state["US Government Social Security Spending"][self.world.timestep] 
                
                self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep + 1] \
                    = self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep] 
                
                self.world.global_state["US Government Income Security"][self.world.timestep + 1] \
                    = self.world.global_state["US Government Income Security"][self.world.timestep]  
                    
                daily_statewise_subsidy_t = self.world.global_state["Subsidy"][curr_t]
                postsubsidy_productivity_t = productivity_t \
                                             + daily_statewise_subsidy_t * \
                                                self.us_government_spending_economic_multiplier
                self.world.global_state["Postsubsidy Productivity"][
                    curr_t
                ] = postsubsidy_productivity_t 
                self.world.global_state["US Treasury Yield Long Term"] = self._real_world_data["us_treasury_yield_long_10_years"][self.world.timestep][0] / 100 if self._real_world_data["us_treasury_yield_long_10_years"][self.world.timestep][0] != 0 else \
                    self.world.global_state["US Treasury Yield Long Term"] 
                self.world.global_state["US Debt"] = self._real_world_data["debt"][self.world.timestep][0] if self._real_world_data["debt"][self.world.timestep][0] != 0 else \
                    self.world.global_state["US Debt"] 
                self.world.global_state["Inflation"] = self._real_world_data["inflation"][self.world.timestep - 1][0] / 100 if int(self._real_world_data["inflation"][self.world.timestep - 1][0]) != 0 else \
                    self.world.global_state["Inflation"]
                theYearIndex = int(self.world.timestep / 365)
                if theYearIndex == 1:
                    self.world.global_state["US GDP"] = self.gdp_2020
                if theYearIndex == 2:
                    self.world.global_state["US GDP"] = self.gdp_2021
                if theYearIndex == 3:
                    self.world.global_state["US GDP"] = self.gdp_2022
            # Current GDP after calculating the death rate 
            # Federal tax revenue 
            else:
                # Considered the tax revenue is attached with the productivity
                self.world.global_state["US Government Revenue"][self.world.timestep] = self.world.global_state["US GDP"] * self.world.global_state["US Tax Wedge"] / 365
                federal_tax_revenue = self.world.global_state["US Government Revenue"][self.world.timestep]
                # Subsidies
                # ---------
                # Add federal government subsidy to productivity
                daily_statewise_subsidy_t = self.world.global_state["Subsidy"][curr_t]
                # postsubsidy_productivity_t = productivity_t * (1 - self.world.global_state["Reduced GDP Multiplier"][self.world.timestep]) + \
                
                postsubsidy_productivity_t = productivity_t + \
                                            daily_statewise_subsidy_t * \
                                                self.us_government_spending_economic_multiplier
                self.world.global_state["Postsubsidy Productivity"][
                    curr_t
                ] = postsubsidy_productivity_t 
                print("Day ", self.world.timestep)
                print("Postsubsidy Productivity ", np.sum(self.world.global_state["Postsubsidy Productivity"][:self.world.timestep], axis=(0, 1)))
                federal_interest_payment = self.world.global_state["US Debt"] * self.world.global_state["US Treasury Yield Long Term"] / 365
                if len(self.world.planner.state["Federal Reserve Balance Sheet"]) > 1 and self.world.planner.state["Federal Reserve Balance Sheet"][-1] != 0:
                    quantitative_amount = self.world.planner.state["Federal Reserve Balance Sheet"][-1]
                    federal_interest_payment = ((self.world.global_state["US Debt"] - quantitative_amount) * self.world.global_state["US Treasury Yield Long Term"] + quantitative_amount * self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep] / 100) / 365
                
                self.world.global_state["US Federal Interest Payment"][self.world.timestep] = federal_interest_payment
                deficit = self.world.global_state["US Government Defense Spending"][self.world.timestep] \
                                                                + self.world.global_state["US Government Social Security Spending"][self.world.timestep] \
                                                                + self.world.global_state["US Government Medicare Medicaid Spending"][self.world.timestep] \
                                                                + self.world.global_state["US Government Income Security"][self.world.timestep] \
                                                                + np.sum(daily_statewise_subsidy_t) + federal_interest_payment - federal_tax_revenue
                self.world.global_state["US Federal Deficit"] += deficit
                self.world.global_state["US Federal Surplus"] += (deficit - federal_interest_payment)
                if self.world.timestep + 1 <= self.episode_length:
                    self.world.global_state["US Debt"] += deficit
                    if self.world.global_state["US Debt"] <= 0:
                        self.world.global_state["US Debt"] = 0
                REAL_POTENTIAL_GDP_2020_2023 = [22168, 23088, 24043, 25015]
                lengthYearCount = len(REAL_POTENTIAL_GDP_2020_2023)
                theYearIndex = int(self.world.timestep / 365)
                theQuarter = 120
                if self.world.timestep % 120 == 0 and self.world.timestep > 0 and self.world.timestep + 1 <= self.episode_length and theYearIndex <= lengthYearCount - 1: 
                    self.world.global_state["US Government Social Security Beneficiaries"] = \
                        self.world.global_state["US Government Social Security Beneficiaries"] + self.social_security_beneficiaries_growth
                    self.world.global_state["US Medicare Medicaid Beneficiaries"] = \
                        self.world.global_state["US Medicare Medicaid Beneficiaries"] + self.medicare_medicaid_beneficiaries_growth
                    self.world.global_state["US Income Security Beneficiaries"] = \
                        self.world.global_state["US Income Security Beneficiaries"] + self.world.planner.state["Total Unemployed"]

                    current_real_potential_gdp = REAL_POTENTIAL_GDP_2020_2023[theYearIndex - 1] * 10**9
                    bet = 0.99 # 
                    omeg = 0.9 # 0.7 in draft, but larger illustrates long term debt effects better. maturity coeff
                    alph = 0.2 # 
                    sig = 0.5 #
                    kap = 0.5 #
                    # kap = 100000 # to produce flex price row of table. Over writes graphs!
                    rhoi = 0.7 #
                    rhos = 0.5 
                    # rho = 0.39 #rho = 1  works too
                    rho = 0.99
                    t_ix = 0.5 # these are set to zero below in the case of no rules 
                    t_ipi = 0.8
                    t_sx = 1
                    t_spi = 0.25
                    b_i = 0
                    b_s= 0 
                    # initialize so functions have an argument. Should not be used before defined. 
                    policy_rules = 1
                    # horizon of simulation is the episode length 

                    fraction_inflated = 0.4 # ratio of sum omega^j pi_j to sum rho^j u_j in fiscal shock. determines b_s
                    
                    previousYearSurplus = 0
                    shock_determinator = -1 if self.world.global_state["US Federal Surplus"] > 0 else 1
                    fiscal_shock = shock_determinator * (self.world.global_state["US Federal Surplus"] - previousYearSurplus) / self.world.global_state["US GDP"]
                    monetary_shock = -(self.world.global_state["Federal Reserve Balance Sheet"] - self.fed_reserve_balance_sheet) / self.world.global_state["US GDP"]
                    if(self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep] != self.fed_fund_rates):
                        monetary_shock += (self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep] - self.fed_fund_rates) / 1 * (self.interest_hikes_shock_gdp / 100)
                    self.fed_reserve_balance_sheet = self.world.global_state["Federal Reserve Balance Sheet"]
                    self.fed_fund_rates = self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep]
                    
                    # horizon of the function is 2 - previous quarter and current quarter, H in the original model is representing year, so in this
                    # implementation, we divided by 4 to measure quarterly impact
                    H = 2
                    
                    
                    monetary_shock_from_previous_year = 0
                    fiscal_shock_from_previous_year = 0
                    monetary_shock += monetary_shock_from_previous_year
                    fiscal_shock += fiscal_shock_from_previous_year
                    shock = [monetary_shock, fiscal_shock] # fiscal shock. 0.01 is 1% of GDP.
                    quarterCount = self.world.timestep / 120
                    if policy_rules == 0: 
                        t_ix = 0
                        t_ipi = 0
                        b_s = 0
                        b_i = 0
                    if policy_rules == 1:
                        b_s_guess = np.array([0, 1])
                        f = lambda b_s: self.parameterfun_s(sig, kap, bet, omeg, rho, t_ix, t_ipi, rhoi, rhos, 
                                                    b_i, b_s, H, t_spi, t_sx, alph, [0, shock[1]], fraction_inflated)
                        # passive policies for both monetary and fiscal, cause the AI will be the one decides these policies
                        f_i = lambda b_i: self.parameterfun(sig, kap, bet, omeg, rho, t_ix, t_ipi, rhoi, rhos, b_i, 0, H, t_spi, t_sx, alph, 
                                                    [shock[0], 0],)
                        b_s, info, ier, msg = fsolve(f, b_s_guess, full_output=True)
                        b_s = np.mean(b_s) 
                        if np.abs(fiscal_shock) == 0:
                            b_s = 0
                        
                        b_i, info, ier, msg = fsolve(f_i, b_s_guess, full_output=True)
                        b_i = np.mean(b_i)
                        if np.abs(monetary_shock) == 0:
                            b_i = 0
                    [N, Nb , nb, Q, ze, Lb] = self.solveFiscalTheoryModel(sig, kap, bet, omeg, rho, t_ix, t_ipi, rhoi, rhos, b_i, b_s, \
                        inflation = self.world.global_state["Inflation"], yieldBond = self.world.global_state["US Treasury Yield Long Term"], outputGap = self.world.global_state["Output Gap"]
                    )
                    
                    [zt, yt, xt, pit, vt, qt, uit, ust, it, st, qlevelt, yldt, rnt,sumomeg,sumratio] =\
                    self.f_doir_final(H , Nb, nb, N, Q, ze, Lb, t_ipi , t_ix , t_spi, t_sx, alph ,omeg , b_s , b_i , shock, rho); 
                    
                    
                    [zt, yt, xt, pit, vt, qt, uit, ust, it, st, qlevelt, yldt, rnt,sumomeg,sumratio] = \
                        self.f_doir_final(H , Nb, nb, N, Q, ze, Lb, t_ipi , t_ix , t_spi, t_sx, alph ,omeg , b_s , b_i , shock, rho)
                    # Divided these results by four, as the FTPL model is using H as year, and we only want to check the quarter impact
                    self.world.global_state["US Treasury Yield Long Term"] = yldt[1] / 4
                    self.world.global_state["Inflation"] =  pit[1] / 4
                    self.world.global_state["Output Gap"] = xt[1] / 4
                    
            # Update agent state
            # ------------------
            current_date_string = datetime.strftime(
                self.current_date, format=self.date_format
            )
            for agent in self.world.agents:
                agent.state["Total Susceptible"] = _S_t[agent.idx].astype(
                    self.np_int_dtype
                )
                agent.state["New Infections"] = (
                    _I_t[agent.idx] - agent.state["Total Infected"]
                ).astype(self.np_int_dtype)
                agent.state["Total Infected"] = _I_t[agent.idx].astype(
                    self.np_int_dtype
                )
                agent.state["Total Recovered"] = _R_t[agent.idx].astype(
                    self.np_int_dtype
                )
                agent.state["New Deaths"] = _D_t[agent.idx] - agent.state[
                    "Total Deaths"
                ].astype(self.np_int_dtype)
                agent.state["Total Deaths"] = _D_t[agent.idx].astype(self.np_int_dtype)
                agent.state["Total Vaccinated"] = _V_t[agent.idx].astype(
                    self.np_int_dtype
                )

                agent.state["Total Unemployed"] = num_unemployed_t[agent.idx].astype(
                    self.np_int_dtype
                )
                agent.state["New Subsidy Received"] = daily_statewise_subsidy_t[
                    agent.idx
                ]
                agent.state["Postsubsidy Productivity"] = postsubsidy_productivity_t[
                    agent.idx
                ]
                agent.state["Date"] = current_date_string

            # Update planner state
            # --------------------
            self.world.planner.state["Total Susceptible"] = np.sum(_S_t).astype(
                self.np_int_dtype
            )
            self.world.planner.state["New Infections"] = (
                np.sum(_I_t) - self.world.planner.state["Total Infected"]
            ).astype(self.np_int_dtype)
            self.world.planner.state["Total Infected"] = np.sum(_I_t).astype(
                self.np_int_dtype
            )
            self.world.planner.state["Total Recovered"] = np.sum(_R_t).astype(
                self.np_int_dtype
            )
            self.world.planner.state["New Deaths"] = (
                np.sum(_D_t) - self.world.planner.state["Total Deaths"]
            ).astype(self.np_int_dtype)
            self.world.planner.state["Total Deaths"] = np.sum(_D_t).astype(
                self.np_int_dtype
            )
            self.world.planner.state["Total Vaccinated"] = np.sum(_V_t).astype(
                self.np_int_dtype
            )
            self.world.planner.state["Total Unemployed"] = np.sum(
                num_unemployed_t
            ).astype(self.np_int_dtype)
            self.world.planner.state["New Subsidy Provided"] = np.sum(
                daily_statewise_subsidy_t
            )
            self.world.planner.state["Postsubsidy Productivity"] = np.sum(
                postsubsidy_productivity_t
            )
            self.world.planner.state["Date"] = current_date_string

    def generate_observations(self):
        """
        - Process agent-specific and planner-specific data into an observation.
        - Observations contain only the relevant features for that actor.
        :return: a dictionary of observations for each agent and planner
        """
        redux_agent_global_state = None
        for feature in [
            "Susceptible",
            "Infected",
            "Recovered",
            "Deaths",
            "Vaccinated",
            "Unemployed",
        ]:
            if redux_agent_global_state is None:
                redux_agent_global_state = self.world.global_state[feature][
                    self.world.timestep
                ]
            else:
                redux_agent_global_state = np.vstack(
                    (
                        redux_agent_global_state,
                        self.world.global_state[feature][self.world.timestep],
                    )
                )
        normalized_redux_agent_state = (
            redux_agent_global_state / self.us_state_population[None]
        )

        # Productivity
        postsubsidy_productivity_t = self.world.global_state[
            "Postsubsidy Productivity"
        ][self.world.timestep] 
        
        normalized_postsubsidy_productivity_t = (
            postsubsidy_productivity_t / self.maximum_productivity_t
        )

        # Let agents know about the policy about to affect SIR infection-rate beta
        t_beta = self.world.timestep - self.beta_delay + 1
        if t_beta < 0:
            lagged_stringency_level = self._real_world_data["policy"][
                self.start_date_index + t_beta
            ]
        else:
            lagged_stringency_level = self.world.global_state["Stringency Level"][
                t_beta
            ]

        normalized_lagged_stringency_level = (
            lagged_stringency_level / self.num_stringency_levels
        )

        # To condition policy on agent id
        agent_index = np.eye(self.n_agents, dtype=self.np_int_dtype)

        # Observation dict - Agents
        # -------------------------
        obs_dict = dict()
        for i in range(self.n_agents):
            # if i == 50:
            #     continue
            obs_dict[str(i)] = {
                "agent_index": agent_index,
                "agent_state": normalized_redux_agent_state,
                "agent_postsubsidy_productivity": normalized_postsubsidy_productivity_t,
                "lagged_stringency_level": normalized_lagged_stringency_level,
            }

        # Observation dict - Planner
        obs_dict[self.world.planner.idx] = { 
            "agent_index": agent_index,
            "agent_state": normalized_redux_agent_state,
            "agent_postsubsidy_productivity": normalized_postsubsidy_productivity_t,
            "lagged_stringency_level": normalized_lagged_stringency_level,
        }

        return obs_dict

    # Heritage Foundation Defense Index - 2020
    # Taxation level and Public Happiness
    # Output Gap should be close to 0
    # Inflation should be close to 0
    # US GDP Growth Rate should be higher than 0
    # Healthcare efficency via spending  
    def compute_reward(self):
        """
        Compute the social welfare metrics for each agent and the planner.
        :return: a dictionary of rewards for each agent in the simulation
        """
        if self.use_cuda:
            self.cuda_compute_reward(
                self.cuda_data_manager.device_data(f"{_REWARDS}_a"),
                self.cuda_data_manager.device_data(f"{_REWARDS}_p"),
                self.cuda_data_manager.device_data("num_days_in_an_year"),
                self.cuda_data_manager.device_data("value_of_life"),
                self.cuda_data_manager.device_data("risk_free_interest_rate"), 
                self.cuda_data_manager.device_data("economic_reward_crra_eta"),
                self.cuda_data_manager.device_data("min_marginal_agent_health_index"),
                self.cuda_data_manager.device_data("max_marginal_agent_health_index"),
                self.cuda_data_manager.device_data("min_marginal_agent_economic_index"),
                self.cuda_data_manager.device_data("max_marginal_agent_economic_index"),
                self.cuda_data_manager.device_data("min_marginal_planner_health_index"),
                self.cuda_data_manager.device_data("max_marginal_planner_health_index"),
                self.cuda_data_manager.device_data(
                    "min_marginal_planner_economic_index"
                ),
                self.cuda_data_manager.device_data(
                    "max_marginal_planner_economic_index"
                ),
                self.cuda_data_manager.device_data(
                    "weightage_on_marginal_agent_health_index"
                ),
                self.cuda_data_manager.device_data(
                    "weightage_on_marginal_agent_economic_index"
                ),
                self.cuda_data_manager.device_data(
                    "weightage_on_marginal_planner_health_index"
                ),
                self.cuda_data_manager.device_data(
                    "weightage_on_marginal_planner_economic_index"
                ),
                self.cuda_data_manager.device_data("agents_health_norm"),
                self.cuda_data_manager.device_data("agents_economic_norm"),
                self.cuda_data_manager.device_data("planner_health_norm"),
                self.cuda_data_manager.device_data("planner_economic_norm"),
                self.cuda_data_manager.device_data("deaths"),
                self.cuda_data_manager.device_data("subsidy"),
                self.cuda_data_manager.device_data("postsubsidy_productivity"),
                self.cuda_data_manager.device_data("_done_"),
                self.cuda_data_manager.device_data("_timestep_"),
                self.cuda_data_manager.device_data("quantitative"),
                self.cuda_data_manager.device_data("USDebt"),
                self.cuda_data_manager.device_data("FederalReserveFundRate"),
                self.cuda_data_manager.meta_info("n_agents"),
                self.cuda_data_manager.meta_info("episode_length"),
                block=self.world.cuda_function_manager.block,
                grid=self.world.cuda_function_manager.grid,
            )
            return {}  # Return empty dict. Reward arrays are updated in-place
        # rew = {"a": 0, "p": 0}
        rew = {}
        for agent in self.world.agents:
            rew[agent.idx] = 0
        rew[self.world.planner.idx] = 0

        def crra_nonlinearity(x, eta):
            # Reference: https://en.wikipedia.org/wiki/Isoelastic_utility
            # To be applied to (marginal) economic indices
            annual_x = self.num_days_in_an_year * x
            annual_x_clipped = np.clip(annual_x, 0.1, 3)
            annual_crra = 1 + (annual_x_clipped ** (1 - eta) - 1) / (1 - eta)
            daily_crra = annual_crra / self.num_days_in_an_year
            return daily_crra

        def min_max_normalization(x, min_x, max_x):
            eps = 1e-10
            return (x - min_x) / (max_x - min_x + eps)

        def get_weighted_average(
            health_index_weightage,
            health_index,
            economic_index_weightage,
            economic_index,
        ):
            return (
                health_index_weightage * health_index
                + economic_index_weightage * economic_index
            ) / (health_index_weightage + economic_index_weightage)

        # Changes this last timestep:
        marginal_deaths = (
            self.world.global_state["Deaths"][self.world.timestep]
            - self.world.global_state["Deaths"][self.world.timestep - 1]
        )

        subsidy_t = self.world.global_state["Subsidy"][self.world.timestep]
        quantitative_t = self.world.global_state["Quantitative"][self.world.timestep]
        postsubsidy_productivity_t = self.world.global_state["Postsubsidy Productivity"][self.world.timestep]
        USDefenseSpending = self.world.global_state["US Government Defense Spending"] 
        USSocialSecuritySpending = self.world.global_state["US Government Social Security Spending"]
        USMedicareMedicaidSpending = self.world.global_state["US Government Medicare Medicaid Spending"]
        USIncomeSecurity = self.world.global_state["US Government Income Security"]
        US_Inflation = self.world.global_state["Inflation"]

        
        # Health index -- the cost equivalent (annual GDP) of covid deaths
        # Note: casting deaths to float to prevent overflow issues
        marginal_agent_health_index = (
            -marginal_deaths.astype(self.np_float_dtype)
            * self.value_of_life
            / self.agents_health_norm
        ).astype(self.np_float_dtype)

        # Economic index -- fraction of annual GDP achieved
        # Use a "crra" nonlinearity on the agent economic reward
        marginal_agent_economic_index = crra_nonlinearity(
            postsubsidy_productivity_t / self.agents_economic_norm,
            self.economic_reward_crra_eta,
        ).astype(self.np_float_dtype)

        # Min-max Normalization
        marginal_agent_health_index = min_max_normalization(
            marginal_agent_health_index,
            self.min_marginal_agent_health_index,
            self.max_marginal_agent_health_index,
        ).astype(self.np_float_dtype)
        marginal_agent_economic_index = min_max_normalization(
            marginal_agent_economic_index,
            self.min_marginal_agent_economic_index,
            self.max_marginal_agent_economic_index,
        ).astype(self.np_float_dtype)

        # Agent Rewards
        # -------------
        agent_rewards = get_weighted_average(
            self.weightage_on_marginal_agent_health_index,
            marginal_agent_health_index,
            self.weightage_on_marginal_agent_economic_index,
            marginal_agent_economic_index,
        )
        # rew["a"] = agent_rewards / self.reward_normalization_factor
        for agent in self.world.agents:
            rew[agent.idx] = agent_rewards[int(agent.idx)] / self.reward_normalization_factor

        # Update agent states
        # -------------------
        for agent in self.world.agents:
            agent.state["Health Index"] += marginal_agent_health_index[agent.idx]
            agent.state["Economic Index"] += marginal_agent_economic_index[agent.idx]

        # National level
        # --------------
        # Health index -- the cost equivalent (annual GDP) of covid deaths
        # Note: casting deaths to float to prevent overflow issues
        marginal_planner_health_index = (
            -np.sum(marginal_deaths).astype(self.np_float_dtype)
            * self.value_of_life
            / self.planner_health_norm
        )

        
        cost_of_subsidy_t = 0
        if (self.world.timestep > 0):
            cost_of_subsidy_t = (1 + self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep]) *\
                                np.sum(quantitative_t) + (np.sum(subsidy_t) - np.sum(quantitative_t)) * \
                                (1 + self.world.global_state["US Treasury Yield Long Term"]) + (self.world.global_state["US GDP"] * self.world.global_state["US Tax Wedge"] / 365) * 0.1
        us_defense_spending_2019 = self.us_government_defense_spending * 365 
        us_imperialism_level_score = 0
        income_security_poverty_reduction_score = 0
        social_security_poverty_reduction_score = 0 
        medicare_medicaid_poverty_reduction_score = 0
        inflation_score = 0
        us_treasury_yield_long_term_score = 0
        other_planner_rewards = 0
        theYearIndex = int(self.world.timestep / 365) if int(self.world.timestep / 365) >= 1 else 0
        getFirstIndexForEveryYear = 365 * (theYearIndex - 1) + 1 if theYearIndex >= 1 else 1
        if self.world.timestep % 365 == 0 and self.world.timestep > 0:
            if not self.use_real_world_data:
                us_imperialism_level_score = np.sum(USDefenseSpending[getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365]) / self.max_us_imperialism_level_spending_required * \
                    self.max_us_imperialism_level
                if us_imperialism_level_score > self.max_us_imperialism_level: 
                    us_imperialism_level_score = self.max_us_imperialism_level
                income_security_poverty_reduction = np.sum(USIncomeSecurity[getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365]) / self.world.global_state["US Income Security Beneficiaries"]
                income_security_poverty_reduction_score = income_security_poverty_reduction/(self.income_security_benefits_avg * (1 + US_Inflation))
                medicare_medicaid_poverty_reduction = np.sum(USMedicareMedicaidSpending[getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365]) / self.world.global_state["US Medicare Medicaid Beneficiaries"]
                medicare_medicaid_poverty_reduction_score = medicare_medicaid_poverty_reduction/(self.medicare_medicaid_benefits_avg * (1 + US_Inflation))
                social_security_poverty_reduction = np.sum(USSocialSecuritySpending[getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365]) / self.world.global_state["US Government Social Security Beneficiaries"]
                social_security_poverty_reduction_score = social_security_poverty_reduction/(self.social_security_benefits_avg * (1 + US_Inflation))
                self.world.planner.state["Defense Index"] += us_imperialism_level_score
                self.world.planner.state["Income Security Index"] += income_security_poverty_reduction_score
                self.world.planner.state["Social Security Index"] += social_security_poverty_reduction_score
                self.world.planner.state["Medicare Medicaid Index"] += medicare_medicaid_poverty_reduction_score
            self.world.planner.state["Inflation Index"] += inflation_score
            self.world.planner.state["US Treasury Yield Index"] += us_treasury_yield_long_term_score

        # Use a "crra" nonlinearity on the planner economic reward
        marginal_planner_economic_index = crra_nonlinearity(
            (np.sum(postsubsidy_productivity_t) - cost_of_subsidy_t) * (1 - US_Inflation)
            / self.planner_economic_norm,
            self.economic_reward_crra_eta,
        )

        # Min-max Normalization
        marginal_planner_health_index = min_max_normalization(
            marginal_planner_health_index,
            self.min_marginal_planner_health_index,
            self.max_marginal_planner_health_index,
        )
        marginal_planner_economic_index = min_max_normalization(
            marginal_planner_economic_index,
            self.min_marginal_planner_economic_index,
            self.max_marginal_planner_economic_index,
        )

        # Update planner states
        # -------------------
        self.world.planner.state["Health Index"] += marginal_planner_health_index
        self.world.planner.state["Economic Index"] += marginal_planner_economic_index
        # Planner Reward
        # --------------
        planner_rewards = get_weighted_average(
            self.weightage_on_marginal_planner_health_index,
            marginal_planner_health_index,
            self.weightage_on_marginal_planner_economic_index,
            marginal_planner_economic_index,
        )
        other_planner_rewards = us_imperialism_level_score + income_security_poverty_reduction_score \
            + social_security_poverty_reduction_score + medicare_medicaid_poverty_reduction_score + inflation_score
        rew[self.world.planner.idx] = (planner_rewards + other_planner_rewards) / (self.reward_normalization_factor) 
        if self.csv_validation and self.world.timestep % 365 != 0 and self.world.timestep >= 30:
            if(os.path.exists(self.csv_file_path) and os.path.isfile(self.csv_file_path)) and self.delete_csv_file: 
                os.remove(self.csv_file_path) 
            self.delete_csv_file = False
            with open(self.csv_file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                # Write the header only once, assuming this script may be run multiple times
                if file.tell() == 0:
                    writer.writeheader()
                
                # Assuming this is inside your simulation loop
                data = {
                    "Month": self.world.timestep / 30,
                    "Susceptibles": np.sum(self.world.global_state["Susceptible"][self.world.timestep]),
                    "Infected": np.sum(self.world.global_state["Infected"][self.world.timestep]),
                    "Recovered": np.sum(self.world.global_state["Recovered"][self.world.timestep]),
                    "Vaccinated (% of population)": np.sum(self.world.global_state["Vaccinated"][self.world.timestep], axis=0) / self.us_population * 100,
                    "Deaths (thousands)": np.sum(self.world.global_state["Deaths"][self.world.timestep], axis=0) / 1e3,
                    "Mean Unemployment Rate (%)": self.world.planner.state["Total Unemployed"] / self.us_population * 100,
                    "US Debt (USD)": self.world.global_state["US Debt"],
                    "US GDP (USD)": self.world.global_state["US GDP"],
                    "Post-productivity (trillion $)": np.sum(self.world.global_state["Postsubsidy Productivity"][getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365], axis=(0, 1)) / 1e12,
                    "Current Subsidy Quantitative Policy Level": self.world.planner.state["Current Subsidy Quantitative Policy Level"],
                    "Total Subsidies (USD)": self.world.planner.state["Total Subsidy"],
                    "US Tax Wedge ('%' of GDP)": self.world.global_state["US Tax Wedge"] * 100,
                    "US Federal Deficit (USD)": (self.world.global_state["US Federal Deficit"]),
                    "US Federal Interest Payment (USD)": np.sum(self.world.global_state["US Federal Interest Payment"][1:]),
                    "Federal Reserve Fund Rate (%)": self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep],
                    "US Treasury Yield Long Term (%)": self.world.global_state["US Treasury Yield Long Term"] * 100,
                    "US Government Revenue (USD)": np.sum(self.world.global_state["US Government Revenue"][getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365]),
                    "US Health Index": self.world.planner.state["Health Index"],
                    "Defense Spending (USD)": np.sum(USDefenseSpending[getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365]),
                    "Income Security Spending (USD)": np.sum(USIncomeSecurity[getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365]),
                    "Social Security Spending (USD)": np.sum(USSocialSecuritySpending[getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365]),
                    "Medicare Medicaid Spending (USD)": np.sum(USMedicareMedicaidSpending[getFirstIndexForEveryYear:getFirstIndexForEveryYear - 1 + 365]),
                    "Federal Reserve Balance Sheet (USD)": self.world.global_state["Federal Reserve Balance Sheet"],
                    "Inflation": US_Inflation,
                    "US Treasury Yield": self.world.global_state["US Treasury Yield Long Term"],
                    "Mean Stringency Level": np.mean(
                        self.world.global_state["Stringency Level"][1:, agent.idx], axis=0
                    ), 
                    "Reward": rew[self.world.planner.idx],
                    "Reward Social Welfare": planner_rewards
                }
                
                # Write the data for this timestep or run
                writer.writerow(data)
    
        # Days
        if self.csv_validation:
            if(os.path.exists(self.csv_file_path_day) and os.path.isfile(self.csv_file_path_day)) and self.delete_csv_day_file: 
                os.remove(self.csv_file_path_day) 
            self.delete_csv_day_file = False
            with open(self.csv_file_path_day, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=headers_day)
                
                # Write the header only once, assuming this script may be run multiple times
                if file.tell() == 0:
                    writer.writeheader()
                
                # Assuming this is inside your simulation loop
                data = {
                    "Day": self.world.timestep,
                    "Susceptibles": np.sum(self.world.global_state["Susceptible"][self.world.timestep]),
                    "Infected": np.sum(self.world.global_state["Infected"][self.world.timestep]),
                    "Recovered": np.sum(self.world.global_state["Recovered"][self.world.timestep]),
                    "Vaccinated (% of population)": np.sum(self.world.global_state["Vaccinated"][self.world.timestep], axis=0) / self.us_population,
                    "Deaths (thousands)": np.sum(self.world.global_state["Deaths"][self.world.timestep], axis=0) / 1e3,
                    "Mean Unemployment Rate (%)": self.world.planner.state["Total Unemployed"] / self.us_population,
                    "US Debt (USD)": self.world.global_state["US Debt"],
                    "US GDP (USD)": self.world.global_state["US GDP"],
                    "Post-productivity (trillion $)": np.sum(self.world.global_state["Postsubsidy Productivity"][1:], axis=(0, 1)),
                    "Current Subsidy Quantitative Policy Level": self.world.planner.state["Current Subsidy Quantitative Policy Level"],
                    "Total Subsidies (USD)": self.world.planner.state["Total Subsidy"],
                    "US Tax Wedge ('%' of GDP)": self.world.global_state["US Tax Wedge"],
                    "US Federal Deficit (USD)":  (self.world.global_state["US Federal Deficit"]),
                    "US Federal Interest Payment (USD)": np.sum(self.world.global_state["US Federal Interest Payment"][1:]) ,
                    "US Government Revenue (USD)": np.sum(self.world.global_state["US Government Revenue"]),
                    "Defense Spending (USD)": np.sum(USDefenseSpending),
                    "Income Security Spending (USD)": np.sum(USIncomeSecurity),
                    "Social Security Spending (USD)": np.sum(USSocialSecuritySpending),
                    "Medicare Medicaid Spending (USD)": np.sum(USMedicareMedicaidSpending),
                    "Federal Reserve Balance Sheet (USD)": self.world.global_state["Federal Reserve Balance Sheet"],
                    "Federal Reserve Fund Rate (%)": self.world.global_state["Federal Reserve Fund Rate"][self.world.timestep],
                    "Inflation": US_Inflation,
                    "US Treasury Yield Long Term (%)": self.world.global_state["US Treasury Yield Long Term"],
                    "Mean Stringency Level": np.mean(
                        self.world.global_state["Stringency Level"][1:, agent.idx], axis=0
                    ),
                    "Reward": rew[self.world.planner.idx],
                    "Reward Social Welfare": planner_rewards
                }
                
                # Write the data for this timestep or run
                writer.writerow(data)
    
        return rew

    def additional_reset_steps(self):
        assert self.world.timestep == 0

        # Reset current date
        self.current_date = self.start_date

        # SIR numbers at timestep 0
        susceptible_0 = self._real_world_data["susceptible"][self.start_date_index]
        infected_0 = self._real_world_data["infected"][self.start_date_index]
        newly_infected_0 = (
            infected_0
            - self._real_world_data["infected"][max(0, self.start_date_index - 1)]
        )
        recovered_0 = self._real_world_data["recovered"][self.start_date_index]
        deaths_0 = recovered_0 * self.death_rate

        # Unemployment and vaccinated numbers at timestep 0
        unemployed_0 = self._real_world_data["unemployed"][self.start_date_index]
        vaccinated_0 = self._real_world_data["vaccinated"][self.start_date_index]

        # Create a global state dictionary to save episode data
        self.world.global_state = {}
        self.dictionary_fiscal_theory = []
        self.set_global_state("Susceptible", susceptible_0, t=self.world.timestep)
        self.set_global_state("Infected", infected_0, t=self.world.timestep)
        self.set_global_state("Recovered", recovered_0, t=self.world.timestep)
        self.set_global_state("Deaths", deaths_0, t=self.world.timestep)

        self.set_global_state("Unemployed", unemployed_0, t=self.world.timestep)
        self.set_global_state("Vaccinated", vaccinated_0, t=self.world.timestep)

        new_deaths_0 = (
            deaths_0
            - self._real_world_data["recovered"][max(0, self.start_date_index - 1)]
            * self.death_rate
        )

        # Reset stringency level history.
        # Pad with stringency levels of 1 corresponding to states being fully open
        # (as was the case before the pandemic).
        self.stringency_level_history = np.pad(
            self._real_world_data["policy"][: self.start_date_index + 1],
            [(self.filter_len, 0), (0, 0)],
            constant_values=1,
        )[-(self.filter_len + 1) :]

        # Set the stringency level based to the real-world policy
        self.set_global_state(
            "Stringency Level",
            self._real_world_data["policy"][self.start_date_index],
            t=self.world.timestep,
        )

        # All US states start with zero subsidy and zero Postsubsidy Productivity
        self.set_global_state("Output Gap", dtype=self.np_float_dtype , value=self.cbo_output_gap_2019, planner=True)
        self.set_global_state("US Debt", dtype=self.np_float_dtype , value=self.us_government_debt, planner=True)
        self.set_global_state("Federal Reserve Balance Sheet", value=self.fed_reserve_balance_sheet, planner=True)
        self.set_global_state("Federal Reserve Fund Rate", value=self.fed_fund_rates, planner=True, isArray=True)
        self.set_global_state("Inflation", value=self.inflation_cpi_2019, planner=True)
        self.set_global_state("US Tax Wedge", value=self.us_tax_wedge, dtype=self.np_float_dtype, planner=True)
        self.set_global_state("US GDP", value=self.us_gdp_2019, planner=True)
        self.set_global_state("US Government Revenue", value=self.us_government_revenue, 
                              dtype=self.np_float_dtype, planner=True, isArray = True)
        self.set_global_state("US Government Mandatory and Discretionary Spending", 
                              value=self.us_government_mandatory_and_discretionary_spending,
                              dtype=self.np_float_dtype, planner=True) 

        self.set_global_state("US Federal Interest Payment", value=self.us_federal_net_interest,
                                dtype=self.np_float_dtype, planner=True, isArray=True)
        self.set_global_state("US Treasury Yield Long Term", value=self.us_treasury_yield_long_term, planner=True)

        self.set_global_state("US Government Defense Spending", value=self.us_government_defense_spending, dtype=self.np_float_dtype, 
                              planner=True, isArray = True)
        self.set_global_state("US Government Social Security Spending", value=self.us_government_social_security_spending,
                               dtype=self.np_float_dtype, planner=True, isArray = True)
        self.set_global_state("US Government Medicare Medicaid Spending", value=self.us_government_medicare_medicaid_spending, 
                              dtype=self.np_float_dtype, planner=True, isArray = True)
        self.set_global_state("US Government Income Security", value=self.us_government_income_security, dtype=self.np_float_dtype, 
                              planner=True, isArray = True)
        self.set_global_state("US Government Social Security Beneficiaries", value=self.social_security_beneficiaries,
                               dtype=self.np_float_dtype, planner=True)
        self.set_global_state("US Medicare Medicaid Beneficiaries", value=self.medicare_medicaid_participants,
                                dtype=self.np_int_dtype, planner=True)
        self.set_global_state("US Income Security Beneficiaries", value=self.income_security_participants,
                                dtype=self.np_int_dtype, planner=True)
        
        self.set_global_state("US Federal Deficit", value=self.us_federal_deficit,
                              dtype=self.np_float_dtype, planner=True)
        self.set_global_state("US Federal Surplus", value=0.0,
                              dtype=self.np_float_dtype, planner=True)
        
        self.set_global_state("Social Security Poverty Reduction", value=self.social_security_poverty_reduction,
                              dtype=self.np_int_dtype, planner=True)
        self.set_global_state("Medicare Medicaid Poverty Reduction", value=self.medicare_medicaid_poverty_reduction,
                              dtype=self.np_int_dtype, planner=True)
        self.set_global_state("Income Security Poverty Reduction", value=self.income_security_poverty_reduction,
                              dtype=self.np_int_dtype, planner=True)
        self.set_global_state("Average Stringency Level", value=0, planner=True)
        # Reduced GDP Multiplier, 0.0 means no reduction
        self.set_global_state("Reduced GDP Multiplier", value=0.0,
                              dtype=self.np_float_dtype, planner=True, isArray=True)
        
        self.set_global_state("Quantitative", dtype=self.np_float_dtype)
        self.set_global_state("Subsidy", dtype=self.np_float_dtype)
        self.set_global_state("Postsubsidy Productivity", dtype=self.np_float_dtype)
        # Set initial agent states
        # ------------------------
        current_date_string = datetime.strftime(
            self.current_date, format=self.date_format
        )

        for agent in self.world.agents:
            agent.state["Total Susceptible"] = susceptible_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["New Infections"] = newly_infected_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["Total Infected"] = infected_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["Total Recovered"] = recovered_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["New Deaths"] = new_deaths_0[agent.idx].astype(
                self.np_int_dtype
            )
            agent.state["Total Deaths"] = deaths_0[agent.idx].astype(self.np_int_dtype)
            agent.state["Health Index"] = np.array([0]).astype(self.np_float_dtype)
            agent.state["Economic Index"] = np.array([0]).astype(self.np_float_dtype)
            agent.state["Date"] = current_date_string

        # Planner state fields
        self.world.planner.state["Total Susceptible"] = np.sum(
            [agent.state["Total Susceptible"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["New Infections"] = np.sum(
            [agent.state["New Infections"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Infected"] = np.sum(
            [agent.state["Total Infected"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Recovered"] = np.sum(
            [agent.state["Total Recovered"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["New Deaths"] = np.sum(
            [agent.state["New Deaths"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Deaths"] = np.sum(
            [agent.state["Total Deaths"] for agent in self.world.agents]
        ).astype(self.np_int_dtype)
        self.world.planner.state["Total Vaccinated"] = np.sum(vaccinated_0).astype(
            self.np_int_dtype
        )
        self.world.planner.state["Health Index"] = np.array([0]).astype(
            self.np_float_dtype
        )
        self.world.planner.state["Economic Index"] = np.array([0]).astype(
            self.np_float_dtype
        )

        self.world.planner.state["Income Security Index"] = np.array([0]).astype(
            self.np_float_dtype
        )
        self.world.planner.state["Social Security Index"] = np.array([0]).astype(
            self.np_float_dtype
        )
        self.world.planner.state["Medicare Medicaid Index"] = np.array([0]).astype(
            self.np_float_dtype
        )
        self.world.planner.state["Defense Index"] = np.array([0]).astype(
            self.np_float_dtype
        )
        self.world.planner.state["Inflation Index"] = np.array([0]).astype(
            self.np_float_dtype
        )
        self.world.planner.state["US Treasury Yield Index"] = np.array([0]).astype(
            self.np_float_dtype
        )
        self.world.planner.state["Date"] = current_date_string

        # Reset any manually set parameter modulations
        self._beta_intercepts_modulation = 1
        self._beta_slopes_modulation = 1
        self._unemployment_modulation = 1

    def set_global_state(self, key=None, value=None, t=None, dtype=None, planner = False, isArray = False):
        # Use floats by default for the SIR dynamics
        if dtype is None:
            dtype = self.np_float_dtype
        if key not in self.world.global_state:
            if planner:
                if isArray:
                    self.world.global_state[key] = np.zeros(
                        (self.episode_length + 1), dtype=dtype
                    )
                    self.world.global_state[key][0] = value
                    self.world.global_state[key][1] = value
                else:
                    self.world.global_state[key] = value
            else:
                self.world.global_state[key] = np.zeros(
                    (self.episode_length + 1, self.num_us_states), dtype=dtype
                )

        if t is not None and value is not None:
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == self.world.global_state[key].shape[1]

            self.world.global_state[key][t] = value
        else:
            pass

    def set_parameter_modulations(
        self, beta_intercept=None, beta_slope=None, unemployment=None
    ):
        """
        Apply parameter modulation, which will be in effect until the next env reset.

        Each modulation term scales the associated set of model parameters by the
        input value. This method is useful for performing a sensitivity analysis.

        In effect, the transmission rate (beta) will be calculated as:
            beta = (m_s * beta_slope)*lagged_stringency + (m_i * beta_intercept)

        The unemployment rate (u) will be calculated as:
            u = SOFTPLUS( m_u * SUM(u_filter_weight * u_filter_response) ) + u_0

        Args:
             beta_intercept: (float, >= 0) Modulation applied to the intercept term
             of the beta model, m_i in above equations
             beta_slope: (float, >= 0) Modulation applied to the slope term of the
             beta model, m_s in above equations
             unemployment: (float, >= 0) Modulation applied to the weighted sum of
             unemployment filter responses, m_u in above equations.

        Example:
            # Reset the environment
            env.reset()

            # Increase the slope of the beta response by 15%
            env.set_parameter_modulations(beta_slope=1.15)

            # Run the environment (this example skips over action selection for brevity)
            for t in range(env.episode_length):
                env.step(actions[t])
        """
        if beta_intercept is not None:
            beta_intercept = float(beta_intercept)
            assert beta_intercept >= 0
            self._beta_intercepts_modulation = beta_intercept

        if beta_slope is not None:
            beta_slope = float(beta_slope)
            assert beta_slope >= 0
            self._beta_slopes_modulation = beta_slope

        if unemployment is not None:
            unemployment = float(unemployment)
            assert unemployment >= 0
            self._unemployment_modulation = unemployment

    def unemployment_step(self, current_stringency_level):
        """
        Computes unemployment given the current stringency level and past levels.

        Unemployment is computed as follows:
        1) For each of self.num_filters, an exponentially decaying filter is
        convolved with the history of stringency changes. Responses move forward in
        time, so a stringency change at time t-1 impacts the response at time t.
        2) The filter responses at time t (the current timestep) are summed together
        using state-specific weights.
        3) The weighted sum is passed through a SOFTPLUS function to capture excess
        unemployment due to stringency policy.
        4) The excess unemployment is added to a state-specific baseline unemployment
        level to get the total unemployment.

        Note: Internally, unemployment is computed somewhat differently for speed.
            In particular, no convolution is used. Instead the "filter response" at
            time t is just a temporally discounted sum of past stringency changes,
            with the discounting given by the filter decay rate.
        """

        def softplus(x, beta=1, threshold=20):
            """
            Numpy implementation of softplus. For reference, see
            https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
            """
            return 1 / beta * np.log(1 + np.exp(beta * x)) * (
                beta * x <= threshold
            ) + x * (beta * x > threshold)

        if (
            self.world.timestep == 0
        ):  # computing unemployment at closure policy "all ones"
            delta_stringency_level = np.zeros((self.filter_len, self.num_us_states))
        else:
            self.stringency_level_history = np.concatenate(
                (
                    self.stringency_level_history[1:],
                    current_stringency_level.reshape(1, -1),
                )
            )
            delta_stringency_level = (
                self.stringency_level_history[1:] - self.stringency_level_history[:-1]
            )

        # Rather than modulating the unemployment params,
        # modulate the deltas (same effect)
        delta_stringency_level = delta_stringency_level * self._unemployment_modulation

        # Expand the [time, state] delta history to have a dimension for filter channel
        x_data = delta_stringency_level[None].transpose(2, 0, 1)

        # Apply the state-specific filter weights to each channel
        weighted_x_data = x_data * self.repeated_conv_weights

        # Compute the discounted sum of the weighted deltas, with each channel using
        # a discounting rate reflecting the time constant of the filter channel. Also
        # sum over channels and use a softplus to get excess unemployment.
        excess_unemployment = softplus(
            np.sum(weighted_x_data * self.unemp_conv_filters, axis=(1, 2)), beta=1
        )

        # Add excess unemployment to baseline unemployment
        unemployment_rate = excess_unemployment + self.unemployment_bias

        # Convert the rate (which is a percent) to raw numbers for output
        num_unemployed_t = unemployment_rate * self.us_state_population / 100
        return num_unemployed_t

    # --- Scenario-specific ---
    def economy_step(
        self,
        population,
        infected,
        deaths,
        unemployed,
        infection_too_sick_to_work_rate=0.05,
        population_between_age_18_65=0.67,
    ):
        """
        Computes how much production occurs.

        Assumptions:

        - People that cannot work: "infected + aware" and "unemployed" and "deaths".
        - No life/death cycles.

        See __init__() for pre-computation of each worker's daily productivity.
        """

        incapacitated = (infection_too_sick_to_work_rate * infected) + deaths
        cant_work = (incapacitated * population_between_age_18_65) + unemployed

        num_workers = population * population_between_age_18_65


        num_people_that_can_work = np.maximum(0, num_workers - cant_work)
        government_spending_per_capita = 0
        if self.use_real_world_data is False and self.world.timestep > 1:
            government_spending = (self.world.global_state["US Government Defense Spending"][self.world.timestep] + self.world.global_state["US Government Social Security Spending"][self.world.timestep] + self.world.global_state["US Government Income Security"][self.world.timestep]) * self.us_government_spending_economic_multiplier
            government_spending_per_capita = government_spending / self.us_population / self.workers_per_capita
        
        self.gdp_per_worker = (self.gdp_per_capita / self.workers_per_capita).astype(
            self.np_float_dtype
        )   
        self.daily_production_per_worker = (
            self.gdp_per_worker / self.num_days_in_an_year
        ).astype(self.np_float_dtype) + government_spending_per_capita
        productivity = (
            num_people_that_can_work * self.daily_production_per_worker
        ).astype(self.np_float_dtype)

        return productivity

    def sir_step(self, S_tm1, I_tm1, stringency_level_tmk, num_vaccines_available_t):
        """
        Simulates SIR infection model in the US.
        """
        
        # A quick guide to the coronavirus variants - https://www.yalemedicine.org/news/covid-19-variants-of-concern-omicron
        covid_variant_factor_alpha = 1.4
        covid_variant_factor_delta = 1.8
        covid_variant_factor_omnicron = 5.6
        
        vaccine_effectiveness_pfizer = [0.94, 0.94, 0.87, 0.87, 0.87, 0.96, 0.96, 0.89, 0.89, 0.89, 0.73, 0.57, 0.57, 0.48]
        # January 2021 - vaccine arrived
        # June 2021 - Delta, November 2021 - Omicron
        # 94% among adults <2months after 2nd dose, pre-Delta period
        # 96% among adults <2 months after 2nd dose, Delta period
        # 87% among adults 4-5 months after 2nd dose, pre-Delta period
        # 89% among adults 4-5 months after 2nd dose, Delta period
        # 73% among adults <2 months after 2nd dose, Omicron period
        # 57% among adults 4-5 months after 2nd dose, Omicron period
        # 96% among adults <2 months after 3rd dose, Delta period
        # 89% among adults <2 months after 3rd dose, Omicron period
        # 66% among adults 4-5 months after 3rd dose, Omicron period
        # 72% among adults ages 50-64 years after 4th dose, Omicron period
        # 76% among adults ages 65 years and older after 4th dose, Omicron period
        # 48% among immunocompromised adults after 4th dose, Omicron period
        
        # 11 months - 330 days after March 2020, we have the covid vaccine in January 2021
        # 7 months - 218 days after March 2020, alpha variant arrived and disappeard in the next 2 months
        # 12 months - 350 days after March 2020, delta variant arrived in March 2021 and disappeard in December 2021
        # 21 months - 613 days after March 2020, omnicron variant arrived
        vaccine_effectiveness = 1
            
        
        intercepts = self.beta_intercepts * self._beta_intercepts_modulation
        slopes = self.beta_slopes * self._beta_slopes_modulation
        beta_i = (intercepts + slopes * stringency_level_tmk).astype(
            self.np_float_dtype
        )
        small_number = 1e-10  # used to prevent indeterminate cases
        susceptible_fraction_vaccinated = np.minimum(
            np.ones((self.num_us_states), dtype=self.np_int_dtype),
            num_vaccines_available_t * vaccine_effectiveness / (S_tm1 + small_number),
        ).astype(self.np_float_dtype)
        vaccinated_t = np.minimum(num_vaccines_available_t, S_tm1) * vaccine_effectiveness

        # Record R0
        R0 = beta_i / self.gamma
        for agent in self.world.agents:
            agent.state["R0"] = R0[agent.idx]

        # S -> I; dS
        neighborhood_SI_over_N = (S_tm1 / self.us_state_population) * I_tm1
        dS_t = (
            -beta_i * neighborhood_SI_over_N * (1 - susceptible_fraction_vaccinated)
            - vaccinated_t * vaccine_effectiveness
        ).astype(self.np_float_dtype)

        # I -> R; dR
        dR_t = (self.gamma * I_tm1 + vaccinated_t * vaccine_effectiveness).astype(self.np_float_dtype)

        # dI from d(S + I + R) = 0
        # ------------------------
        dI_t = -dS_t - dR_t

        dV_t = vaccinated_t.astype(self.np_float_dtype)

        return dS_t, dI_t, dR_t, dV_t

    def load_model_constants(self, path_to_model_constants):
        filename = "model_constants.json"
        assert filename in os.listdir(path_to_model_constants), (
            "Unable to locate '{}' in '{}'.\nPlease run the "
            "'gather_real_world_data.ipynb' notebook first".format(
                filename, path_to_model_constants
            )
        )
        with open(os.path.join(path_to_model_constants, filename), "r") as fp:
            model_constants_dict = json.load(fp)
        fp.close()

        self.date_format = model_constants_dict["DATE_FORMAT"]
        self.us_state_idx_to_state_name = model_constants_dict[
            "US_STATE_IDX_TO_STATE_NAME"
        ]
        self.us_state_population = self.np_int_dtype(
            model_constants_dict["US_STATE_POPULATION"]
        )
        self.us_population = self.np_int_dtype(model_constants_dict["US_POPULATION"])
        self.num_stringency_levels = model_constants_dict["NUM_STRINGENCY_LEVELS"]
        self.death_rate = self.np_float_dtype(model_constants_dict["SIR_MORTALITY"])
        self.gamma = self.np_float_dtype(model_constants_dict["SIR_GAMMA"])
        self.gdp_per_capita = self.np_float_dtype(
            model_constants_dict["GDP_PER_CAPITA"]
        )

    def load_fitted_params(self, path_to_fitted_params):
        filename = "fitted_params.json"
        assert filename in os.listdir(path_to_fitted_params), (
            "Unable to locate '{}' in '{}'.\nIf you ran the "
            "'gather_real_world_data.ipynb' notebook to download the latest "
            "real-world data, please also run the "
            "'fit_parameters.ipynb' notebook.".format(filename, path_to_fitted_params)
        )
        with open(os.path.join(path_to_fitted_params, filename), "r") as fp:
            fitted_params_dict = json.load(fp)
        fp.close()
        self.policy_start_date = datetime.strptime(
            fitted_params_dict["POLICY_START_DATE"], self.date_format
        )
        self.value_of_life = self.np_int_dtype(fitted_params_dict["VALUE_OF_LIFE"])
        self.beta_delay = self.np_int_dtype(fitted_params_dict["BETA_DELAY"])
        self.beta_slopes = np.array(
            fitted_params_dict["BETA_SLOPES"], dtype=self.np_float_dtype
        )
        self.beta_intercepts = np.array(
            fitted_params_dict["BETA_INTERCEPTS"], dtype=self.np_float_dtype
        )
        self.min_marginal_agent_health_index = np.array(
            fitted_params_dict["MIN_MARGINAL_AGENT_HEALTH_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.max_marginal_agent_health_index = np.array(
            fitted_params_dict["MAX_MARGINAL_AGENT_HEALTH_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.min_marginal_agent_economic_index = np.array(
            fitted_params_dict["MIN_MARGINAL_AGENT_ECONOMIC_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.max_marginal_agent_economic_index = np.array(
            fitted_params_dict["MAX_MARGINAL_AGENT_ECONOMIC_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.min_marginal_planner_health_index = self.np_float_dtype(
            fitted_params_dict["MIN_MARGINAL_PLANNER_HEALTH_INDEX"]
        )
        self.max_marginal_planner_health_index = self.np_float_dtype(
            fitted_params_dict["MAX_MARGINAL_PLANNER_HEALTH_INDEX"]
        )
        self.min_marginal_planner_economic_index = self.np_float_dtype(
            fitted_params_dict["MIN_MARGINAL_PLANNER_ECONOMIC_INDEX"]
        )
        self.max_marginal_planner_economic_index = self.np_float_dtype(
            fitted_params_dict["MAX_MARGINAL_PLANNER_ECONOMIC_INDEX"]
        )
        self.inferred_weightage_on_agent_health_index = np.array(
            fitted_params_dict["INFERRED_WEIGHTAGE_ON_AGENT_HEALTH_INDEX"],
            dtype=self.np_float_dtype,
        )
        self.inferred_weightage_on_planner_health_index = self.np_float_dtype(
            fitted_params_dict["INFERRED_WEIGHTAGE_ON_PLANNER_HEALTH_INDEX"]
        )
        self.filter_len = self.np_int_dtype(fitted_params_dict["FILTER_LEN"])
        self.conv_lambdas = np.array(
            fitted_params_dict["CONV_LAMBDAS"], dtype=self.np_float_dtype
        )
        self.unemployment_bias = np.array(
            fitted_params_dict["UNEMPLOYMENT_BIAS"], dtype=self.np_float_dtype
        )
        self.grouped_convolutional_filter_weights = np.array(
            fitted_params_dict["GROUPED_CONVOLUTIONAL_FILTER_WEIGHTS"],
            dtype=self.np_float_dtype,
        )

    def scenario_metrics(self):
        # End of episode metrics
        # ----------------------
        metrics_dict = {}

        # State-level metrics
        for agent in self.world.agents:
            state_name = self.us_state_idx_to_state_name[str(agent.idx)]

            for field in ["infected", "recovered", "deaths"]:
                metric_key = "{}/{} (millions)".format(state_name, field)
                metrics_dict[metric_key] = (
                    agent.state["Total " + field.capitalize()] / 1e6
                )

            metrics_dict["{}/mean_unemployment_rate (%)".format(state_name)] = (
                np.mean(self.world.global_state["Unemployed"][1:, agent.idx], axis=0)
                / self.us_state_population[agent.idx]
                * 100
            )

            metrics_dict[
                "{}/mean_open_close_stringency_level".format(state_name)
            ] = np.mean(
                self.world.global_state["Stringency Level"][1:, agent.idx], axis=0
            )

            metrics_dict["{}/total_productivity (billion $)".format(state_name)] = (
                np.sum(
                    self.world.global_state["Postsubsidy Productivity"][1:, agent.idx]
                )
                / 1e9
            )

            metrics_dict[
                "{}/health_index_at_end_of_episode".format(state_name)
            ] = agent.state["Health Index"]
            metrics_dict[
                "{}/economic_index_at_end_of_episode".format(state_name)
            ] = agent.state["Economic Index"]

        # USA-level metrics
        metrics_dict["usa/vaccinated (% of population)"] = (
            np.sum(self.world.global_state["Vaccinated"][self.world.timestep], axis=0)
            / self.us_population
            * 100
        )
        metrics_dict["usa/deaths (thousands)"] = (
            np.sum(self.world.global_state["Deaths"][self.world.timestep], axis=0) / 1e3
        )

        metrics_dict["usa/mean_unemployment_rate (%)"] = (
            np.mean(
                np.sum(self.world.global_state["Unemployed"][1:], axis=1)
                / self.us_population,
                axis=0,
            )
            * 100
        )
        metrics_dict["usa/total_amount_subsidized (trillion $)"] = (
            np.sum(self.world.global_state["Subsidy"][1:], axis=(0, 1)) / 1e12
        )
        metrics_dict["usa/total_productivity (trillion $)"] = (
            np.sum(self.world.global_state["Postsubsidy Productivity"][1:], axis=(0, 1))
            / 1e12
        )

        metrics_dict["usa/health_index_at_end_of_episode"] = self.world.planner.state[
            "Health Index"
        ]
        metrics_dict["usa/economic_index_at_end_of_episode"] = self.world.planner.state[
            "Economic Index"
        ]

        return metrics_dict
 

    
    def solveFiscalTheoryModel(self, sig, kap, bet, omeg, rho, t_ix, t_ipi, rhoi, rhos, b_i, b_s, inflation = 0, yieldBond = 0, outputGap = 0):
        # matrices
        show_results = False  # use for debugging
        A = np.eye(5)  
        # print(A.shape[0])
        N = A.shape[0]
    

        # x pi q ui us
        # [0.988, 0.026, 0, 0, 0]
        B = np.array([
            [1 + sig * t_ix + sig * kap / bet, sig * t_ipi - sig / bet, 0, sig, 0],
            [-kap / bet, 1 / bet, 0, 0, 0],
            [t_ix / omeg, t_ipi / omeg, 1 / omeg, 1 / omeg, 0],
            [0, 0, 0, rhoi, 0],
            [0, 0, 0, 0, rhos]
        ])

        C = np.array([
            [0, 0],
            [-b_i, -b_s],
            [0, 0],
            [1, 0],
            [0, 1]
        ])

        D = np.array([
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0]
        ])

        # Solve by eigenvalues
        A1 = np.linalg.inv(A)
        F = np.dot(A1, B)
        L, Q = np.linalg.eig(F)  
        Q1 = np.linalg.inv(Q)
        if show_results:
            print('Eigenvalues')
            print(np.abs(np.diag(L)))

        # produce Ef, Eb, that select forward and backward eigvenvalues.
        nf = np.where(np.abs(L) >= 1)[0]  # nf is the index of eigenvalues greater than one
        if show_results:
            print('number of eigenvalues >=1')
            print(nf.shape[0])
        if nf.shape[0] < D.shape[1]:
            print('not enough eigenvalues greater than 1')
        Ef = np.zeros((nf.shape[0], A.shape[1]))
        Efstar = np.zeros((A.shape[1], A.shape[1]))
        for indx in range(nf.shape[0]):
            Ef[indx, nf[indx]] = 1
            Efstar[nf[indx], nf[indx]] = 1
    
    
        indices = np.abs(L.T)

        # Print the result
        nb = np.where(indices < 1)[0]
        Eb = np.zeros((nb.shape[0], A.shape[1]))
        for indx in range(nb.shape[0]):
            Eb[indx, nb[indx]] = 1


        for indx in range(0, 3):
            if indx == 0:
                if indx in nf:
                    Ef[np.where(nf == (indx))[0], indx] = 1 + outputGap
                    Efstar[indx, indx] = 1 + outputGap
                if indx in nb:
                    Eb[np.where(nb == (indx))[0], indx] = 1 + outputGap
            
            if indx == 1:
                if indx in nf:
                    Ef[np.where(nf == (indx))[0], indx] = 1 + inflation
                    Efstar[indx, indx] = 1 + inflation
                if indx in nb:
                    Eb[np.where(nb == (indx))[0], indx] = 1 + inflation
            
            if indx == 2:
                if indx in nf:
                    Ef[np.where(nf == (indx))[0], indx] = 1 + yieldBond
                    Efstar[indx, indx] = 1 + yieldBond
                if indx in nb:
                    Eb[np.where(nb == (indx))[0], indx] = 1 + yieldBond
            
        
        # ze = np.dot(Eb, np.dot(Q1, np.dot(A1,
        #     C - np.dot(D, np.dot(inverse_linalg_Q1_A1_D,Ef_Q1_A1_C))
        # )))
        ze = np.dot(Eb, Q1)
        ze = np.dot(ze, A1) 
        Ef_Q1 = np.dot(Ef, Q1)
        Ef_Q1_A1 = np.dot(Ef_Q1, A1) 
        Ef_Q1_A1_D = np.dot(Ef_Q1_A1, D)
        D_Ef_Q1_A1_D = np.dot(D, np.linalg.inv(Ef_Q1_A1_D)) 
        D_Ef_Q1_A1_D_Ef = np.dot(D_Ef_Q1_A1_D, Ef)
        D_Ef_Q1_A1_D_Ef_Q1 = np.dot(D_Ef_Q1_A1_D_Ef, Q1)
        D_Ef_Q1_A1_D_Ef_Q1_A1 = np.dot(D_Ef_Q1_A1_D_Ef_Q1, A1)
        D_Ef_Q1_A1_D_Ef_Q1_A1_C = np.dot(D_Ef_Q1_A1_D_Ef_Q1_A1, C) 
        ze = np.dot(ze, C - D_Ef_Q1_A1_D_Ef_Q1_A1_C)

        # how epsilon shocks map to z.
        # in principle the forward z are zero. In practice they are 1E-16 and then
        # grow. So I go through the trouble of simulating forward only the nonzero
        # z and eigenvalues less than one. 
        Nb = Eb.shape[0]  # number of stable z's 
        Lb = np.dot(np.dot(Eb, np.diag(L)), Eb.T)  # diagonal with only stable Ls

        return N, Nb, nb, Q, ze, Lb
    
            
    def f_doir_final(self, H, Nb, nb, N, Q, ze, Lb, t_ipi, t_ix, t_spi, t_sx, alph, omeg, b_s, b_i, shock, rho):
        zbt = np.zeros((H, Nb))
        
        zbt[1, :] = np.dot(ze, shock) 
        for t in range(2, H):
            zbt[t, :] = np.dot(Lb, zbt[t-1, :]) 
        zt = np.zeros((H, N))
        zt[:, nb] = zbt 
        yt = np.dot(zt, Q.T)
        xt = yt[:, 0]
        pit = yt[:, 1]
        qt = yt[:, 2]
        uit = yt[:, 3]
        ust = yt[:, 4]

        rnt = np.zeros(H)
        rnt = omeg * qt - np.concatenate(([0], qt[:-1]))

        vt = np.zeros(H)
        for t in range(1, H):
            vt[t] = 1 / rho * ((1 - alph) * vt[t-1] + rnt[t] - (1 + t_spi) * pit[t] - t_sx * xt[t] - ust[t])

        it = t_ipi * pit + t_ix * xt + uit
        st = t_spi * pit + t_sx * xt + alph * np.concatenate(([0], vt[:-1])) + ust

        qlevelt = qt - np.log(1 - omeg)
        yldt = np.exp(-qlevelt) + omeg - 1

        sumomeg = np.sum((omeg ** np.arange(H-1)) * pit[1:]) * 100
        rhoj = np.concatenate(([0], rho ** np.arange(st.shape[0]-1)))
        sumratio = 0
        if np.sum(rhoj * ust) != 0:
            sumratio = -np.sum((omeg ** np.arange(H-1)) * pit[1:]) / np.sum(rhoj * ust)
        # zt, yt, xt, pit, vt, qt, uit, ust, it, st, qlevelt, yldt, rnt, sumomeg, sumratio
        return zt, yt, xt, pit, vt, qt, uit, ust, it, st, qlevelt, yldt, rnt, sumomeg, sumratio
        
    

    def parameterfun_s(self, sig, kap, bet, omeg, rho, t_ix, t_ipi, rhoi, rhos, b_i, b_s, H, t_spi, t_sx, alph, shock, fraction_inflated):
        sumratio = np.zeros_like(b_s)
        for ind in range(b_s.shape[0]):
            N, Nb, nb, Q, ze, Lb = self.solveFiscalTheoryModel(sig, kap, bet, omeg, rho, t_ix, t_ipi, rhoi, rhos, b_i, b_s[ind])
            zt, yt, xt, pit, vt, qt, uit, ust, it, st, qlevelt, yldt, rnt, sumomeg, sumratio[ind] = self.f_doir_final(H, Nb, nb, N, Q, ze, Lb, t_ipi, t_ix, t_spi, t_sx, alph, omeg, b_s[ind], b_i, shock, rho)
            sumratio[ind] = sumratio[ind] - fraction_inflated
        return sumratio 

    def parameterfun(self, sig, kap, bet, omeg, rho, t_ix, t_ipi, rhoi, rhos, b_i, b_s, H, t_spi, t_sx, alph, shock):
        sumomeg = np.zeros_like(b_i)
        for ind in range(b_i.shape[0]):
            N, Nb, nb, Q, ze, Lb = self.solveFiscalTheoryModel(sig, kap, bet, omeg, rho, t_ix, t_ipi, rhoi, rhos, b_i[ind], b_s)
            zt, yt, xt, pit, vt, qt, uit, ust, it, st, qlevelt, yldt, rnt, sumomeg[ind], sumratio = self.f_doir_final(H, Nb, nb, N, Q, ze, Lb, t_ipi, t_ix, t_spi, t_sx, alph, omeg, b_s, b_i[ind], shock, rho)
        return sumomeg