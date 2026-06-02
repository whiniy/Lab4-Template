import stat
import dataclasses
from agents import EntityAgent, UncertainAgent
from model import Location,GameState, GameAction, WizardMoves, Location, Crystal, Portal, Lava, GameTransitions, Observation, EmptyTile, LocationCounts, LocationDistribution
import random


class MDP:
    def __init__(self,initial_state: GameState, escape_reward:float,living_reward: float,death_reward:float, discount: float):
        self.game_state = initial_state
        self.living_reward = living_reward
        self.death_reward = death_reward
        self.escape_reward = escape_reward
        self.discount = discount

    def reward(self,source:GameState,target:GameState, action: GameAction) -> float:
        loc = target.active_entity_location
        if isinstance(target.tile_grid[loc.row][loc.col],Lava):
            return self.death_reward
        elif target.victory:
            return self.escape_reward
        else:
            return self.living_reward


    def transition_model(self,location: Location, action: GameAction) -> LocationDistribution:
        """
        Transition model of the MDP, gives conditional probability distribution of result location given starting location and action choice.
        """
        source_state = self.game_state.replace_active_entity_location(location)
        successors = GameTransitions.get_successors(source_state)
        actions = [a for a, _ in successors]
        successor_states = [state for _, state in successors]

        if action not in actions:
            return self.transition_model(location,WizardMoves.STAY)

        # Outcomes are either the desired outcome of the action, or a random other action each with 50% prob.
        possible_results = LocationCounts(self.game_state.grid_size)
        for i in range(len(actions)):
            if actions[i] == action:
                for _ in range(len(actions)+1):
                    possible_results.add_count(successor_states[i].active_entity_location)
            else:
                possible_results.add_count(successor_states[i].active_entity_location)

        return possible_results.normalize()

    def transition_distribution(self, source: LocationDistribution, action: GameAction) -> LocationDistribution:
        """
        Given a location distribution, calculate the new distribution that is a result of taking the given action.
        The easiest way to do this will involve sampling.
        """
        # create empty counter for locations
        location_counts = LocationCounts(self.game_state.grid_size)
        # repeat sampling for 1000 iterations 
        for _ in range(1000):
            # get possible location from source distribution
            current_location = source.sample()
            # use transition_model to see where wizard could end up if it takes the action 
            transition_distribution = self.transition_model(current_location, action)
            # sample new location from transition model
            new_location = transition_distribution.sample()
            # add count to new location in location counts
            location_counts.add_count(new_location)
        
        # convert location counts to distribution and return
        return location_counts.normalize()


class LocationValues:
    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.value_grid = [[0.0 for _ in range(mdp.game_state.grid_size[1])] for _ in range(mdp.game_state.grid_size[1])]

    def value_iteration(self,k):
        for _ in range(k):
            self.value_iteration_update()

    def value_iteration_update(self):
        """
        Perform one update of value iteration based off of the provided MDP.
        """

        rows = self.mdp.game_state.grid_size[0]
        cols = self.mdp.game_state.grid_size[1]

        # Fix value_grid size for non-square maps like cliff
        if len(self.value_grid) != rows or len(self.value_grid[0]) != cols:
            self.value_grid = [[0.0 for _ in range(cols)] for _ in range(rows)]

        # New grid for updated values
        next_value_grid = [[0.0 for _ in range(cols)] for _ in range(rows)]

        # Go through every location
        for row in range(rows):
            for col in range(cols):
                location = Location(row, col)

                # Create a state with the wizard at this location
                source_state = self.mdp.game_state.replace_active_entity_location(location)

                # Get actual successor states, including victory/defeat info
                successors = GameTransitions.get_successors(source_state)

                # If no successors, leave value as 0
                if len(successors) == 0:
                    next_value_grid[row][col] = 0
                    continue

                actions = [a for a, _ in successors]
                action_values = []

                # Try each possible action
                for action in WizardMoves:

                    # If action is invalid, transition_model treats it like STAY
                    action_to_use = action
                    if action not in actions:
                        action_to_use = WizardMoves.STAY

                    # Matches transition_model weighting
                    total_weight = 2 * len(actions)
                    expected_value = 0

                    for successor_action, successor_state in successors:

                        # Intended action gets extra probability weight
                        if successor_action == action_to_use:
                            weight = len(actions) + 1
                        else:
                            weight = 1

                        probability = weight / total_weight

                        # IMPORTANT: handle terminal states directly
                        if successor_state.defeat:
                            reward = self.mdp.death_reward
                            future_value = 0
                        elif successor_state.victory:
                            reward = self.mdp.escape_reward
                            future_value = 0
                        else:
                            reward = self.mdp.living_reward
                            next_loc = successor_state.active_entity_location
                            future_value = self.value_grid[next_loc.row][next_loc.col]

                        expected_value += probability * (
                            reward + self.mdp.discount * future_value
                        )

                    action_values.append(expected_value)

                next_value_grid[row][col] = max(action_values)

        self.value_grid = next_value_grid
        return next_value_grid


class MDPAgent(UncertainAgent):
    values: LocationValues
    current_position_estimate: LocationDistribution
    current_score_estimate: int
    mdp: MDP

    def __init__(self, mdp: MDP, value_iteration_steps=100):
        self.mdp = mdp
        self.values = LocationValues(mdp)
        self.values.value_iteration(value_iteration_steps)
        self.current_position_estimate = LocationDistribution.from_game_state_uniform(mdp.game_state)


    def observation_likelihood(self, observation: Observation, loc: Location)-> float:
        portal_loc = self.mdp.game_state.get_all_tile_locations(Portal)[0]
        portal_dist = abs(loc.row-portal_loc.row) + abs(loc.col - portal_loc.col)

        if abs(portal_dist - observation.approximatePortalDist) > 1:
            return 0
        else:
            return 1.0/3.0

    def update_prior(self,action: GameAction):
        self.current_position_estimate = self.mdp.transition_distribution(self.current_position_estimate,action)


    def update_belief(self, observation: Observation):
        """
        Use Bayes rule to update your beliefs about the wizard location by updating self.current_position_estimate.
        You have prior belief of your current estimate P(Loc), and the observation likelihood model (P(Obs | Loc)).
        Use these to calculate the new belief.
        """

        #We need to update our belief for all possible locations. So lets start by creating a new distribution
        new_estimate = LocationDistribution.from_game_state_uniform(self.mdp.game_state)

        #The new distribution should be set for each location
        # P(Loc | Obs) = P(Obs | Loc) * P(Loc)
        for loc in new_estimate.locations():
            # get old probability that wizard is at this location before using the new observation
            prior = self.current_position_estimate.probability(loc)
            
            # get likelihood of the observation given that the wizard is at this location
            likelihood = self.observation_likelihood(observation, loc)
            
            # Bayes' rule
            # new belief is proportional to old belief times likelihood of the observation
            new_prob = likelihood * prior
            
            # store the updated probability for this location in the new distribution
            new_estimate.update_probability(loc, new_prob)

        # normalize the new distribution so that it sums to 1
        new_estimate.renormalize()
        
        # replace old belief with new belief
        self.current_position_estimate = new_estimate

    def react(self, observation: Observation) -> GameAction:
        """
        Our uncertain agent only has noisy observations to guess where in the dungeon it is. Use the previously implemented parts to generate an estimate for the distribution of possible locations the wizard might be at, updating every turn with a new observation, and choose the action based off of your value iteration policy based on your estimate.
        """


        # Part 2: Choose the best action
        # use your calculated value iteration map of location values alongside your estimated location to choose the best action given your uncertain state.
        # There are multiple ways to do this, but some things to consider:
            # 1. You want to select the action which will have the highest expected value, given the probability distribution of the resultant states.
            # 2. The expected value of some quantity is just the weighted average value of that quantity in an outcome weighted by the probability of that outcome over all outcomes (and can be estimated by the average of a sufficiently big sample of outcomes sampled from the distribution)
            # 3. You can sample locations from any distribution, and can form distributions of locations from samples
            # 4. You can find the distribution of the results of an action for a given specific location
            # 5. You can calculate the reward of a specific transition as a result of a specific action with a specific result
            # 6. You have an estimate of the value of each result location
        
        # Update belief based on noisy observation
        self.update_belief(observation)

        best_action = None
        best_value = float("-inf")

        # Try every action
        for action in WizardMoves:
            expected_value = 0

            # Consider every location the wizard might be at
            for curr_loc in self.current_position_estimate.locations():
                current_prob = self.current_position_estimate.probability(curr_loc)

                source_state = self.mdp.game_state.replace_active_entity_location(curr_loc)
                successors = GameTransitions.get_successors(source_state)

                if len(successors) == 0:
                    continue

                actions = [a for a, _ in successors]

                # If action is invalid, transition_model treats it like STAY
                action_to_use = action
                if action not in actions:
                    action_to_use = WizardMoves.STAY

                # Same probability logic as transition_model
                total_weight = 2 * len(actions)

                for successor_action, successor_state in successors:

                    if successor_action == action_to_use:
                        weight = len(actions) + 1
                    else:
                        weight = 1

                    next_prob = weight / total_weight

                    # IMPORTANT: handle lava/portal directly
                    if successor_state.defeat:
                        reward = self.mdp.death_reward
                        future_value = 0
                    elif successor_state.victory:
                        reward = self.mdp.escape_reward
                        future_value = 0
                    else:
                        reward = self.mdp.living_reward
                        next_loc = successor_state.active_entity_location
                        future_value = self.values.value_grid[next_loc.row][next_loc.col]

                    expected_value += current_prob * next_prob * (
                        reward + self.mdp.discount * future_value
                    )

            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        action = best_action

        # Update belief after choosing this action
        self.update_prior(action)

        return action