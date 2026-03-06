# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

# ---------------------------------------------------------------------------
# BASE AGENT
# Shared utilities for both OffensiveAgent and DefensiveAgent:
#   - Precomputes home_boundary (midline cells on our side)
#   - Provides A* search used by both agents
# ---------------------------------------------------------------------------

class PacmanAgent(CaptureAgent):
    """Base class: shared A* search and home-boundary precomputation."""

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        # CaptureAgent.register_initial_state precomputes all pairwise maze
        # distances so get_maze_distance() is O(1) at action time.

        walls = game_state.get_walls()
        mid = walls.width // 2
        # Red owns x < mid, blue owns x >= mid.
        # home_boundary = non-wall cells on the column closest to the enemy.
        home_x = mid - 1 if self.red else mid
        self.home_boundary = [
            (home_x, y) for y in range(1, walls.height - 1)
            if not walls[home_x][y]
        ]

    # -----------------------------------------------------------------------
    # A* SEARCH  (Lecture 5 — Heuristic Search)
    #
    # f(n) = g(n) + h(n)
    #   g(n) = uniform cost (1 per step)
    #   h(n) = min maze_distance(n, goal) for goal in goals
    #          Admissible because maze_distance == true shortest-path cost
    #          (delete-relaxation from Lecture 5).
    #
    # Parameters
    # ----------
    # start : (x, y) starting position
    # goals : iterable of target (x, y) positions (any one counts)
    # avoid : set of (x, y) cells treated as impassable (ghost danger zones)
    #
    # Returns the first Directions action on the optimal path, or None.
    # -----------------------------------------------------------------------
    def astar(self, game_state, start, goals, avoid=None):
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
# OFFENSIVE AGENT  (Steps 1–3)
#
# Classical planning model (Lecture 5, STRIPS):
#   F  = grid positions
#   s0 = current position,  G = food positions,  A = {N, S, E, W}
#   Uses A* with h = min maze_distance to food (admissible).
#
# Extensions:
#   Step 2 — return home after carrying >= threshold pellets (score them)
#   Step 3 — treat cells near visible non-scared ghosts as impassable in A*
# ---------------------------------------------------------------------------

class OffensiveAgent(PacmanAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # Step 3: Manhattan-distance buffer around visible enemy ghosts.
        self.safety_dist = 2

        # Dynamic threshold tuning constants (see _return_threshold below).
        # These replace the old fixed self.return_threshold = 3.
        self.min_carry   = 1   # always return if carrying this many and close
        self.max_carry   = 6   # never carry more than this regardless of distance

    def _return_threshold(self, game_state, my_pos):
        """
        Dynamic carry threshold based on distance to the home boundary.

        Intuition
        ---------
        If we are already near the boundary, it costs little to deposit food,
        so we should return even with just 1–2 pellets.
        If we are deep in enemy territory, the trip home is expensive, so we
        should fill up more before making the journey.

        Formula
        -------
        dist_home  = maze distance to the nearest boundary cell
        max_dist   = width of the enemy half ≈ walls.width // 2

        threshold = min_carry + round((dist_home / max_dist) * (max_carry - min_carry))

        This linearly scales the threshold from min_carry (at the boundary)
        to max_carry (at the far edge of the enemy side).
        """
        dist_home = min(self.get_maze_distance(my_pos, b) for b in self.home_boundary)
        max_dist  = game_state.get_walls().width // 2          # approx enemy half-width
        ratio     = min(dist_home / max_dist, 1.0)             # clamp to [0, 1]
        threshold = self.min_carry + round(ratio * (self.max_carry - self.min_carry))
        return threshold

    def choose_action(self, game_state):
        my_state  = game_state.get_agent_state(self.index)
        my_pos    = my_state.get_position()
        carrying  = my_state.num_carrying
        food_list = self.get_food(game_state).as_list()
        capsules  = self.get_capsules(game_state)

        # Compute dynamic threshold for this position
        threshold = self._return_threshold(game_state, my_pos)

        # Step 3: cells to avoid this turn
        visible_ghosts = self._visible_ghosts(game_state)
        danger = self._danger_positions(game_state)
        ghost_threat = any(
            self.get_maze_distance(my_pos, g.get_position()) <= 5
            for g in visible_ghosts
        )

        # Step 4a: if ghosts are scared, drop all danger avoidance and eat freely.
        if self._scared_ghosts(game_state):
            danger = set()
            ghost_threat = False

        # Step 4b: go for capsule if it's closer than the threatening ghost.
        if ghost_threat and capsules:
            nearest_ghost_dist = min(
                self.get_maze_distance(my_pos, g.get_position()) for g in visible_ghosts
            )
            nearest_capsule_dist = min(
                self.get_maze_distance(my_pos, c) for c in capsules
            )
            if nearest_capsule_dist < nearest_ghost_dist:
                action = self.astar(game_state, my_pos, capsules, avoid=danger)
                if action is not None:
                    return action

        # Step 5: ENDGAME — sprint home if time is nearly up and we are carrying food.
        # Conditions to trigger endgame sprint:
        #   a) carrying any food AND moves left for us <= dist_home + buffer  →
        #      we literally won't make it if we don't leave now
        #   b) carrying food AND moves left for us <= endgame_window  →
        #      close enough to the end that depositing > exploring
        if carrying > 0:
            dist_home    = min(self.get_maze_distance(my_pos, b) for b in self.home_boundary)
            moves_left   = game_state.data.timeleft // 4   # our share of remaining moves
            buffer       = 5                               # safety margin in steps
            endgame_window = 60                            # always go home in last 60 of our moves

            must_go_home = (moves_left <= dist_home + buffer) or \
                           (moves_left <= endgame_window)

            if must_go_home:
                # Don't let danger avoidance block us if there's no time to go around
                action = self.astar(game_state, my_pos, self.home_boundary, avoid=danger)
                if action is None:
                    action = self.astar(game_state, my_pos, self.home_boundary)
                if action is not None:
                    return action

        # Step 2: decide whether to return home (now uses dynamic threshold)
        should_return = (carrying >= threshold) or \
                        (carrying > 0 and len(food_list) <= 2) or \
                        (carrying > 0 and ghost_threat)

        if should_return:
            action = self.astar(game_state, my_pos, self.home_boundary, avoid=danger)
            if action is None:
                action = self.astar(game_state, my_pos, self.home_boundary)
            if action is not None:
                return action

        # Step 1: navigate to food, avoiding danger zone
        if not food_list:
            return random.choice(game_state.get_legal_actions(self.index))

        action = self.astar(game_state, my_pos, food_list, avoid=danger)
        if action is None:
            if carrying > 0:
                action = self.astar(game_state, my_pos, self.home_boundary, avoid=danger) \
                      or self.astar(game_state, my_pos, self.home_boundary)
            else:
                action = self.astar(game_state, my_pos, food_list)

        return action or random.choice(game_state.get_legal_actions(self.index))

    def _visible_ghosts(self, game_state):
        """Visible, non-scared enemy ghosts."""
        ghosts = []
        for idx in self.get_opponents(game_state):
            e = game_state.get_agent_state(idx)
            if not e.is_pacman and e.scared_timer == 0 and e.get_position() is not None:
                ghosts.append(e)
        return ghosts

    def _scared_ghosts(self, game_state):
        """Visible enemy ghosts that are currently scared (safe to ignore)."""
        ghosts = []
        for idx in self.get_opponents(game_state):
            e = game_state.get_agent_state(idx)
            if not e.is_pacman and e.scared_timer > 0 and e.get_position() is not None:
                ghosts.append(e)
        return ghosts

    def _danger_positions(self, game_state):
        """Diamond of cells within safety_dist of each visible ghost."""
        danger = set()
        walls  = game_state.get_walls()
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


# ---------------------------------------------------------------------------
# DEFENSIVE AGENT  (Step 4)
#
# Belief tracking for invisible enemies (Lecture 6 — Partially Observable
# Environments / MDPs with imperfect information):
#
#   Belief  b(s) = P(enemy is at grid cell s)
#
#   Predict step — transition model (enemy takes one step per turn):
#     b̂(s') = Σ_s  P(s' | s) · b(s)
#     P(s' | s) = 1 / |neighbours(s)|   (uniform random move + stay)
#
#   Observe step — Bayes' rule with noisy sonar:
#     b_{t+1}(s) ∝ P(obs | s) · b̂(s)
#     game_state.get_agent_distances() → noisy Manhattan distance
#     P(obs | s) = get_distance_prob(manhattan(me, s), obs)
#               = 1/13  if |obs − true_dist| ≤ 6,  else 0
#
#   Direct observation (enemy within SIGHT_RANGE = 5):
#     b_{t+1}(s) = point mass at the exact observed position
#
# Action selection:
#   - If invader is visible       → A* chase to exact position
#   - If invader is invisible     → A* chase to most-likely cell (arg_max b)
#   - No invaders                 → A* patrol along home boundary
# ---------------------------------------------------------------------------

class DefensiveAgent(PacmanAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.walls = game_state.get_walls()

        # Initialize beliefs as a point mass at each opponent's start position.
        # util.Counter is a dict subclass with normalize() and arg_max().
        self.beliefs = {}
        for opp_idx in self.get_opponents(game_state):
            b = util.Counter()
            b[game_state.get_initial_agent_position(opp_idx)] = 1.0
            self.beliefs[opp_idx] = b

        # Patrol waypoints: upper and lower quarters of the home boundary column.
        # The agent cycles between them so it never stands still while guarding.
        n = len(self.home_boundary)
        self.patrol_points = [
            self.home_boundary[n // 4],
            self.home_boundary[3 * n // 4],
        ]
        self.patrol_idx = 0

    # ------------------------------------------------------------------
    # Belief update: predict then observe (called at the start of each turn)
    # ------------------------------------------------------------------

    def _predict(self):
        """Transition update: spread probability mass to adjacent cells."""
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
        Bayesian update using the noisy sonar readings.

        For each opponent:
          - If directly visible  → collapse belief to exact position (point mass).
          - Otherwise            → weight each candidate position by the sonar
                                   likelihood and renormalize.
        """
        my_pos     = game_state.get_agent_position(self.index)
        noisy_dists = game_state.get_agent_distances()

        for opp_idx in self.get_opponents(game_state):
            opp_state  = game_state.get_agent_state(opp_idx)
            exact_pos  = opp_state.get_position()

            if exact_pos is not None:
                # Direct observation: collapse to point mass
                b = util.Counter()
                b[exact_pos] = 1.0
                self.beliefs[opp_idx] = b
            else:
                noisy = noisy_dists[opp_idx]
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
                # else: keep old belief (numerical edge case — all weights zero)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, game_state):
        # Update belief state every turn
        self._predict()
        self._observe(game_state)

        my_state = game_state.get_agent_state(self.index)
        my_pos   = my_state.get_position()

        # Don't leave our side — if we accidentally crossed, come back.
        if my_state.is_pacman:
            action = self.astar(game_state, my_pos, self.home_boundary)
            return action or random.choice(game_state.get_legal_actions(self.index))

        # Gather invader targets (visible = exact pos; invisible = most-likely pos)
        targets = []
        for opp_idx in self.get_opponents(game_state):
            opp = game_state.get_agent_state(opp_idx)
            if not opp.is_pacman:
                continue  # still on enemy side, not our problem yet
            if opp.get_position() is not None:
                targets.append(opp.get_position())
            else:
                targets.append(self.beliefs[opp_idx].arg_max())

        if targets:
            action = self.astar(game_state, my_pos, targets)
            if action:
                return action

        # No invaders: cycle between two patrol waypoints along the boundary.
        # When the current waypoint is reached, advance to the next one.
        target = self.patrol_points[self.patrol_idx]
        if my_pos == target:
            self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_points)
            target = self.patrol_points[self.patrol_idx]
        action = self.astar(game_state, my_pos, [target])
        return action or random.choice(game_state.get_legal_actions(self.index))