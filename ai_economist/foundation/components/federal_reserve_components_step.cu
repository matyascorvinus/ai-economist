// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

extern "C" {
    // CUDA version of the components in
    // "ai_economist.foundation.components.covid19_components.py"
 
    __global__ void CudaFederalQuantitativeEasingStep(
        int * QE_level,
        float * QE,
        const int kQEInterval,
        const int kNumQELevels,
        const float * KMaxDailyQEPerState,
        const int * kDefaultFederalReserveActionMask,
        const int * kNoOpFederalReserveActionMask,
        int * actions,
        float * obs_a_time_until_next_QE,
        float * obs_a_current_QE_level,
        float * obs_f_time_until_next_QE,
        float * obs_f_current_QE_level,
        float * obs_f_action_mask,
        int * env_timestep_arr,
        const int kNumAgents,
        const int kEpisodeLength
    ) {
        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;

        assert(env_timestep_arr[kEnvId] > 0 &&
            env_timestep_arr[kEnvId] <= kEpisodeLength);
        assert (kAgentId <= kNumAgents - 1);

        int t_since_last_QE = env_timestep_arr[kEnvId] %
            kQEInterval;

        // Setting the (federal government) Federal Reserve's QE level
        // to be the QE level for all the US states
        if (kAgentId < kNumAgents - 1) {
            // Indices for time-dependent and time-independent arrays
            // Time dependent arrays have shapes (num_envs,
            // kEpisodeLength + 1, kNumAgents - 1)
            // Time independent arrays have shapes (num_envs, kNumAgents - 1)
            const int kArrayIdxOffset = kEnvId * (kEpisodeLength + 1) *
                (kNumAgents - 1);
            int time_dependent_array_index_curr_t = kArrayIdxOffset +
                env_timestep_arr[kEnvId] * (kNumAgents - 1) + kAgentId;
            int time_dependent_array_index_prev_t = kArrayIdxOffset +
                (env_timestep_arr[kEnvId] - 1) * (kNumAgents - 1) + kAgentId;
            const int time_independent_array_index = kEnvId *
                (kNumAgents - 1) + kAgentId;

            if ((env_timestep_arr[kEnvId] - 1) % kQEInterval == 0) {
                assert(0 <= actions[kEnvId] <= kNumQELevels);
                QE_level[time_dependent_array_index_curr_t] =
                    actions[kEnvId];
            } else {
                QE_level[time_dependent_array_index_curr_t] =
                    QE_level[time_dependent_array_index_prev_t];
            }
            // Setting the subsidies for the US states
            // based on the federal government's QE level
            QE[time_dependent_array_index_curr_t] =
                QE_level[time_dependent_array_index_curr_t] *
                KMaxDailyQEPerState[kAgentId] / kNumQELevels;

            obs_a_time_until_next_QE[
                time_independent_array_index] =
                    1 - (t_since_last_QE /
                    static_cast<float>(kQEInterval));
            obs_a_current_QE_level[
                time_independent_array_index] =
                    QE_level[time_dependent_array_index_curr_t] /
                    static_cast<float>(kNumQELevels);
        } else if (kAgentId == (kNumAgents - 1)) {
            for (int action_id = 0; action_id < kNumQELevels + 1;
                action_id++) {
                int action_mask_array_index = kEnvId *
                    (kNumQELevels + 1) + action_id;
                if (env_timestep_arr[kEnvId] % kQEInterval == 0) {
                    obs_f_action_mask[action_mask_array_index] =
                        kDefaultFederalReserveActionMask[action_id];
                } else {
                    obs_f_action_mask[action_mask_array_index] =
                        kNoOpFederalReserveActionMask[action_id];
                }
            }
            
            // Update FederalReserve obs after the agent's obs are updated
            __syncthreads();

            if (kAgentId == (kNumAgents - 1)) {
                // Just use the values for agent id 0
                obs_f_time_until_next_QE[kEnvId] =
                    obs_a_time_until_next_QE[
                        kEnvId * (kNumAgents - 1)
                    ];
                obs_f_current_QE_level[kEnvId] = 
                    obs_a_current_QE_level[
                        kEnvId * (kNumAgents - 1)
                    ];
            }
        }
    }

 
}
