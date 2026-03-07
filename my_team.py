# my_team.py
#
# Our approach has two agents:
#   - OffensiveAgent: navigates to food using A* and decides when to return home based on how many pellets
#   it's carrying and how far it is from the boundary. It also avoids visible ghosts and uses power capsules.
#   - DefensiveAgent: tracks where enemies probably are using a Bayesian belief filter and patrols the
#   boundary near our biggest food cluster.

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions


# ---------------------------------------------------------------------------
# Team creation
# ---------------------------------------------------------------------------

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


# ---------------------------------------------------------------------------
# Base agent
#
# Both agents share two things: the home boundary (the column of non-wall
# cells on our side of the map closest to the enemy) and an A* search
# function used to plan paths to any set of goal positions.
# ---------------------------------------------------------------------------

class PacmanAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # Precompute the home boundary: the column of walkable cells just on our side of the midline.
        # Returning to any of these cells deposits the food we are carrying and scores the points.
        walls = game_state.get_walls()
        mid = walls.width // 2
        home_x = mid - 1 if self.red else mid
        self.home_boundary = [
            (home_x, y) for y in range(1, walls.height - 1)
            if not walls[home_x][y]
        ]

    def astar(self, game_state, start, goals, avoid=None):
        """
        A* search from start to the nearest cell in goals.

        We use maze distance as the heuristic (admissible because it equals the true shortest path on this grid).
        The avoid parameter lets us mark danger zones around ghosts as impassable so the agent routes around them.

        Returns the first action to take, or None if no path exists.
        """
        if avoid is None:
            avoid = set()
        walls = game_state.get_walls()
        goals = set(goals)
        if not goals:
            return None

        def h(pos):
            return min(self.get_maze_distance(pos, g) for g in goals)

        frontier = util.PriorityQueue()
        frontier.push((start, None, 0), h(start))
        best_g = {start: 0}

        moves = [
            (Directions.NORTH, (0,  1)),
            (Directions.SOUTH, (0, -1)),
            (Directions.EAST,  (1,  0)),
            (Directions.WEST,  (-1, 0)),
        ]

        while not frontier.is_empty():
            pos, first_action, g = frontier.pop()

            if pos in goals:
                return first_action if first_action is not None else Directions.STOP

            if g > best_g.get(pos, float('inf')):
                continue

            for action, (dx, dy) in moves:
                nx = int(pos[0] + dx)
                ny = int(pos[1] + dy)
                next_pos = (nx, ny)

                if walls[nx][ny] or next_pos in avoid:
                    continue

                new_g = g + 1
                if new_g < best_g.get(next_pos, float('inf')):
                    best_g[next_pos] = new_g
                    f = new_g + h(next_pos)
                    next_first = action if first_action is None else first_action
                    frontier.push((next_pos, next_first, new_g), f)

        return None


# ---------------------------------------------------------------------------
# Offensive agent
#
# The offensive agent crosses into enemy territory to collect food and bring it back to score.
# The main decisions it makes each turn are:
#
#   1. Should I go home now? — decided by a dynamic threshold: if we are close to the boundary it's cheap to
#      deposit so we return early; if we are deep in enemy territory we collect more first.
#
#   2. Is there a ghost nearby? — if so, we either flee or grab a power capsule if one is closer than the ghost.
#      If ghosts are already scared we ignore them and eat freely.
#
#   3. Is time almost up? — if we won't make it home before the game ends we sprint back immediately so the food
#      we're carrying gets scored.
#
#   4. Otherwise, A* to the nearest food, routing around danger zones.
# ---------------------------------------------------------------------------

class OffensiveAgent(PacmanAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # How many cells around a visible ghost we treat as off-limits
        self.safety_dist = 2

        # Bounds for the dynamic return threshold (see _return_threshold)
        self.min_carry = 1   # return even with 1 pellet if we're near home
        self.max_carry = 6   # never carry more than this no matter how far away

    # --- helpers -----------------------------------------------------------

    def _visible_ghosts(self, game_state):
        """Enemy ghosts that are visible and not currently scared."""
        ghosts = []
        for idx in self.get_opponents(game_state):
            e = game_state.get_agent_state(idx)
            if not e.is_pacman and e.scared_timer == 0 and e.get_position() is not None:
                ghosts.append(e)
        return ghosts

    def _scared_ghosts(self, game_state):
        """Enemy ghosts that are visible and scared (harmless to us)."""
        ghosts = []
        for idx in self.get_opponents(game_state):
            e = game_state.get_agent_state(idx)
            if not e.is_pacman and e.scared_timer > 0 and e.get_position() is not None:
                ghosts.append(e)
        return ghosts

    def _danger_positions(self, game_state):
        """
        Returns the set of cells within safety_dist steps of any visible non-scared ghost.
        We pass this to A* as the avoid set so the agent naturally routes around ghost proximity zones.
        """
        danger = set()
        walls = game_state.get_walls()
        for ghost in self._visible_ghosts(game_state):
            gx, gy = int(ghost.get_position()[0]), int(ghost.get_position()[1])
            for dx in range(-self.safety_dist, self.safety_dist + 1):
                for dy in range(-self.safety_dist, self.safety_dist + 1):
                    if abs(dx) + abs(dy) <= self.safety_dist:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < walls.width and 0 <= ny < walls.height:
                            if not walls[nx][ny]:
                                danger.add((nx, ny))
        return danger

    def _return_threshold(self, game_state, my_pos):
        """
        How many pellets should we carry before going home?

        The answer depends on where we are. Near the boundary a single pellet is worth going back for (the trip
        is almost free). Deep in enemy territory we should fill up more so the long journey home pays off.
        We scale linearly between min_carry and max_carry based on our current maze distance from the boundary.
        """
        dist_home = min(self.get_maze_distance(my_pos, b) for b in self.home_boundary)
        max_dist  = game_state.get_walls().width // 2
        ratio     = min(dist_home / max_dist, 1.0)
        return self.min_carry + round(ratio * (self.max_carry - self.min_carry))

    # --- main decision logic -----------------------------------------------

    def choose_action(self, game_state):
        my_state  = game_state.get_agent_state(self.index)
        my_pos    = my_state.get_position()
        carrying  = my_state.num_carrying
        food_list = self.get_food(game_state).as_list()
        capsules  = self.get_capsules(game_state)

        threshold      = self._return_threshold(game_state, my_pos)
        visible_ghosts = self._visible_ghosts(game_state)
        danger         = self._danger_positions(game_state)

        ghost_threat = any(
            self.get_maze_distance(my_pos, g.get_position()) <= 5
            for g in visible_ghosts
        )

        # If ghosts are scared we can ignore all danger and eat freely
        if self._scared_ghosts(game_state):
            danger = set()
            ghost_threat = False

        # If a ghost is close, go for a capsule if we can reach it first that way we neutralize the threat
        if ghost_threat and capsules:
            nearest_ghost_dist   = min(self.get_maze_distance(my_pos, g.get_position())
                                       for g in visible_ghosts)
            nearest_capsule_dist = min(self.get_maze_distance(my_pos, c)
                                       for c in capsules)
            if nearest_capsule_dist < nearest_ghost_dist:
                action = self.astar(game_state, my_pos, capsules, avoid=danger)
                if action is not None:
                    return action

        # Endgame: go now if we won't make it home before time runs out
        if carrying > 0:
            dist_home      = min(self.get_maze_distance(my_pos, b) for b in self.home_boundary)
            moves_left     = game_state.data.timeleft // 4
            buffer         = 5
            endgame_window = 60

            if moves_left <= dist_home + buffer or moves_left <= endgame_window:
                action = self.astar(game_state, my_pos, self.home_boundary, avoid=danger)
                if action is None:
                    action = self.astar(game_state, my_pos, self.home_boundary)
                if action is not None:
                    return action

        # Return home if we hit our carry threshold, food is almost gone, or a ghost is threatening
        # and we have something worth saving
        should_return = (
            carrying >= threshold or
            (carrying > 0 and len(food_list) <= 2) or
            (carrying > 0 and ghost_threat)
        )

        if should_return:
            action = self.astar(game_state, my_pos, self.home_boundary, avoid=danger)
            if action is None:
                action = self.astar(game_state, my_pos, self.home_boundary)
            if action is not None:
                return action

        # go collect the nearest food, routing around ghost zones
        if not food_list:
            return random.choice(game_state.get_legal_actions(self.index))

        action = self.astar(game_state, my_pos, food_list, avoid=danger)
        if action is None:
            if carrying > 0:
                action = (self.astar(game_state, my_pos, self.home_boundary, avoid=danger)
                          or self.astar(game_state, my_pos, self.home_boundary))
            else:
                action = self.astar(game_state, my_pos, food_list)

        return action or random.choice(game_state.get_legal_actions(self.index))


# ---------------------------------------------------------------------------
# Defensive agent
#
# The defensive agent stays on our side and tries to intercept enemy pacman before they can deposit our food.
# Enemies are only visible within 5 squares and outside that range we only get a noisy distance reading.
# We handle this with a Bayesian belief filter: we maintain a probability distribution over where each enemy
# probably is and update it every turn using the sonar readings.
#
# Each turn:
#   1. Predict: the enemy could have moved one step in any direction, so we spread the probability mass to neighboring cells.
#   2. Observe: weight each cell by how well its distance to us matches the noisy sonar reading we received (Bayes).
#   3. Act: if an enemy is visible, chase directly; if not, head to the most likely cell from our belief distribution;
#   if nobody has crossed yet, patrol the boundary near our biggest food cluster.
# ---------------------------------------------------------------------------

class DefensiveAgent(PacmanAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.walls = game_state.get_walls()

        self.beliefs = {}
        for opp_idx in self.get_opponents(game_state):
            b = util.Counter()
            b[game_state.get_initial_agent_position(opp_idx)] = 1.0
            self.beliefs[opp_idx] = b

        # Fallback patrol position: center of the home boundary column used only if there is no food left to protect
        n = len(self.home_boundary)
        self.patrol_fallback = self.home_boundary[n // 2]

    # --- belief update -----------------------------------------------------

    def _predict(self):
        """
        Transition step: the enemy can move one step in any direction (or stay still), so we spread each cell's
        probability equally among its walkable neighbors.
        """
        walls = self.walls
        step_offsets = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

        for opp_idx, belief in self.beliefs.items():
            new_belief = util.Counter()
            for pos, prob in belief.items():
                if prob == 0:
                    continue
                x, y = int(pos[0]), int(pos[1])
                neighbors = [
                    (x + dx, y + dy)
                    for dx, dy in step_offsets
                    if 0 <= x + dx < walls.width
                    and 0 <= y + dy < walls.height
                    and not walls[x + dx][y + dy]
                ]
                share = prob / len(neighbors)
                for nb in neighbors:
                    new_belief[nb] += share
            new_belief.normalize()
            self.beliefs[opp_idx] = new_belief

    def _observe(self, game_state):
        """
        Observation step: update the belief using Bayes' rule.

        If the enemy is directly visible we collapse the belief to a point mass at their exact position.
        Otherwise, we weight each candidate cell by how likely it is to produce the noisy sonar reading we got
        """
        my_pos      = game_state.get_agent_position(self.index)
        noisy_dists = game_state.get_agent_distances()

        for opp_idx in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp_idx)
            exact_pos = opp_state.get_position()

            if exact_pos is not None:
                b = util.Counter()
                b[exact_pos] = 1.0
                self.beliefs[opp_idx] = b
            else:
                noisy      = noisy_dists[opp_idx]
                new_belief = util.Counter()
                for pos, prior in self.beliefs[opp_idx].items():
                    if prior == 0:
                        continue
                    true_dist  = util.manhattan_distance(my_pos, pos)
                    likelihood = game_state.get_distance_prob(true_dist, noisy)
                    new_belief[pos] = prior * likelihood

                if new_belief.total_count() > 0:
                    new_belief.normalize()
                    self.beliefs[opp_idx] = new_belief
                # else: keep old belief — all likelihoods were zero (edge case) todo


    def _best_patrol_target(self, game_state, my_pos):
        """
        Find the boundary cell that best covers our most threatened food.

        We split our remaining food into two clusters (top half / bottom half of the map) and find which one is larger.
        We pick the boundary cell closest to the center of that cluster since an attacker heading for that food will
        likely cross there, so positioning ourselves there makes it more probable to intercept.
        """
        food_list = self.get_food_you_are_defending(game_state).as_list()
        if not food_list:
            return self.patrol_fallback

        k = min(2, len(food_list))
        sorted_by_y = sorted(food_list, key=lambda f: f[1])
        centres = [sorted_by_y[-1]]
        if k == 2:
            centres.append(sorted_by_y[0])

        clusters = {i: [] for i in range(k)}
        for food in food_list:
            nearest_c = min(range(k),
                            key=lambda i: util.manhattan_distance(food, centres[i]))
            clusters[nearest_c].append(food)

        largest = max(clusters.values(), key=len)
        if not largest:
            return self.patrol_fallback

        # Snap centroid to the nearest real food cell to stay off walls
        cx     = sum(f[0] for f in largest) / len(largest)
        cy     = sum(f[1] for f in largest) / len(largest)
        anchor = min(largest, key=lambda f: (f[0] - cx) ** 2 + (f[1] - cy) ** 2)

        return min(self.home_boundary,
                   key=lambda b: self.get_maze_distance(b, anchor))

    # --- main decision logic -----------------------------------------------

    def choose_action(self, game_state):
        # Update our belief about where enemies are before deciding anything
        self._predict()
        self._observe(game_state)

        my_state = game_state.get_agent_state(self.index)
        my_pos   = my_state.get_position()

        # If we somehow crossed into enemy territory, come straight back
        if my_state.is_pacman:
            action = self.astar(game_state, my_pos, self.home_boundary)
            return action or random.choice(game_state.get_legal_actions(self.index))

        # Build list of invader targets to chase
        targets = []
        for opp_idx in self.get_opponents(game_state):
            opp = game_state.get_agent_state(opp_idx)
            if not opp.is_pacman:
                continue
            if opp.get_position() is not None:
                targets.append(opp.get_position())
            else:
                # Use the most likely cell from our belief distribution
                targets.append(self.beliefs[opp_idx].arg_max())

        if targets:
            action = self.astar(game_state, my_pos, targets)
            if action:
                return action

        # No invaders yet — patrol the boundary near our biggest food cluster
        target = self._best_patrol_target(game_state, my_pos)
        action = self.astar(game_state, my_pos, [target])
        return action or random.choice(game_state.get_legal_actions(self.index))