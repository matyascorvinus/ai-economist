# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from ai_economist.foundation.base.base_agent import BaseAgent, agent_registry


@agent_registry.add
class BasicFederalReserve(BaseAgent):
    """
    A basic Federal Reserve agent represents US Federal Reserve that sets monetary policy.

    Unlike the "mobile" agent, the planner does not represent an embodied agent in
    the world environment. BasicFederalReserve modifies the BaseAgent class to remove
    location as part of the agent state.
    
    This planner will work indenpendently of the BasicPlanner agent. Designed to be used in conjunction with the BasicPlanner agent.
    Designated as 'f'
    """

    name = "BasicFederalReserve"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.state["loc"]

        # Overwrite any specified index so that this one is always indexed as 'p'
        # (make a separate class of planner if you want there to be multiple planners
        # in a game)
        self._idx = "f"

    @property
    def loc(self):
        """
        BasicFederalReserve agents do not occupy any location.
        """
        raise AttributeError("BasicFederalReserve agents do not occupy a location.")
