import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import clear_output
from coordinate_utils import *


class Gridworld:
    def __init__(self,
                 num_rows=5,
                 num_cols=5,
                 num_gold=1,
                 num_bombs=1,
                 gold_positions=np.array([23]),
                 bomb_positions=np.array([18]),
                 gold_reward=10,
                 bomb_reward=-10,
                 random_move_probability=0.02,
                 load_images=True):
        """
        The Gridworld class contains all functionality of the env that is needed for an agent to learn.
        Furthermore, it contains a visualize() method to visualize the current state of the env.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_gold = num_gold
        self.num_bombs = num_bombs
        self.gold_reward = gold_reward
        self.bomb_reward = bomb_reward
        self.gold_positions = gold_positions
        self.bomb_positions = bomb_positions
        self.terminal_states = np.append(self.gold_positions, self.bomb_positions)
        self.random_move_probability = random_move_probability

        self.actions = ["UP", "RIGHT", "DOWN", "LEFT"]
        self.num_actions = len(self.actions)
        self.num_states = self.num_cols * self.num_rows
        self.rewards = np.zeros(shape=self.num_states)
        self.rewards[self.bomb_positions] = self.bomb_reward
        self.rewards[self.gold_positions] = self.gold_reward

        self.step = 0
        self.cumulative_reward = 0
        self.agent_position = 0

        # Visualization parameter
        self.m = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1, vmax=1), cmap=cm.coolwarm)
        if load_images:
            self.gold_image = plt.imread("images/gold.png")
            self.bomb_image = plt.imread("images/bomb.jpg")
            self.agent_image = plt.imread("images/agent_2.png")

        self.reset()


    def reset(self):
        self.agent_position = np.random.randint(0, 5)

    def make_step(self, action_index):
        """
        Given an action, make state transition and observe reward.

        :param action_index: an integer between 0 and the number of actions (4 in Gridworld).
        :return: (reward, new_position)
            WHERE
            reward (float) is the observed reward
            new_position (int) is the new position of the agent
        """
        # Randomly sample action_index if world is stochastic
        if np.random.uniform(0, 1) < self.random_move_probability:
            action_indices = np.arange(self.num_actions, dtype=int)
            action_indices = np.delete(action_indices, action_index)
            action_index = np.random.choice(action_indices, 1)[0]

        action = self.actions[action_index]

        # Determine new position and check whether the agent hits a wall.
        old_position = self.agent_position
        new_position = self.agent_position
        if action == "UP":
            candidate_position = old_position + self.num_cols
            if candidate_position < self.num_states:
                new_position = candidate_position
        elif action == "RIGHT":
            candidate_position = old_position + 1
            if candidate_position % self.num_cols > 0:  # The %-operator denotes "modulo"-division.
                new_position = candidate_position
        elif action == "DOWN":
            candidate_position = old_position - self.num_cols
            if candidate_position >= 0:
                new_position = candidate_position
        elif action == "LEFT":  # "LEFT"
            candidate_position = old_position - 1
            if candidate_position % self.num_cols < self.num_cols - 1:
                new_position = candidate_position
        else:
            raise ValueError('Action was mis-specified!')

        # Update the env state
        self.agent_position = new_position

        # Calculate reward
        reward = self.rewards[self.agent_position]
        reward -= 1
        return reward, new_position

    def is_terminal_state(self):
        # The following statement returns a boolean. It is 'True' when the agent_position
        # coincides with any bomb_positions or gold_positions.
        return self.agent_position in np.append(self.bomb_positions, self.gold_positions)

    def visualize(self, show_grid=True, show_policy=False, show_learning_curve=False,
                  show_q_values=False, clear_the_output=True, show_state_labels=False,
                  episode=0, reward_per_episode=np.zeros(0), agent_q_table=np.zeros((2, 2)), fig_size=5):
        """
        Visualize the grid (with or without policy) and / or the current learning curve.

        :param show_grid:
        :param show_policy:
        :param show_learning_curve:
        :param clear_the_output: if True, the plot is overwritten in each iteration. (Avoids flickering effect in Jupyter)
        :param episode:
        :param reward_per_episode: an array of length `episode`
        :param agent_q_table:
        """
        if clear_the_output:
            clear_output(wait=True)

        # Determine figure structure and size depending on what shall be plotted.
        if show_grid and show_learning_curve:
            fig, (policy_plot, lc_plot) = plt.subplots(figsize=(3*fig_size, fig_size), ncols=2, nrows=1, gridspec_kw={'width_ratios': [1.5, 1]})  #
        elif show_grid:
            fig, policy_plot = plt.subplots(figsize=(fig_size, fig_size))
        elif show_learning_curve:
            fig, lc_plot = plt.subplots(figsize=(fig_size, fig_size * 6 / 10))

        if show_grid:
            policy_plot.set_xlim((0, self.num_cols))
            policy_plot.set_ylim((0, self.num_rows))
            policy_plot.tick_params(length=0, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            plt.grid()
            # Add bombs
            for pos in self.bomb_positions:
                image_coordinates = position_to_image_coordinates(pos, self.num_cols)
                policy_plot.imshow(self.bomb_image, extent=image_coordinates)

            # Add gold
            for pos in self.gold_positions:
                image_coordinates = position_to_image_coordinates(pos, self.num_cols)
                policy_plot.imshow(self.gold_image, extent=image_coordinates)

        if show_q_values:
            for state in range(len(agent_q_table)):
                if state not in self.terminal_states:
                    x, y = one_d_to_two_d(state, self.num_cols)
                    q_values = agent_q_table[state, :]
                    # maximum_q_values = np.nonzero(q_values == np.amax(q_values))[0]
                    num_max = 1  # len(q_values)
                    for dir in np.arange(4):
                        dx, dy = direction_to_x_y(dir)
                        # print(m.to_rgba(q_values[dir]))
                        p = mpatches.Arrow(x + 0.5, y + 0.5, dx / num_max / 2.5, dy / num_max / 2.5,
                                           facecolor=self.m.to_rgba(q_values[dir]), width=0.5)
                        policy_plot.add_patch(p)

        if show_policy:
            for state in range(len(agent_q_table)):
                if state not in self.terminal_states:
                    x, y = one_d_to_two_d(state, self.num_cols)
                    q_values = agent_q_table[state, :]
                    maximum_q_values = np.nonzero(q_values == np.amax(q_values))[0]
                    num_max = len(maximum_q_values)
                    for dir in maximum_q_values:
                        dx, dy = direction_to_x_y(dir)
                        p = mpatches.Arrow(x + 0.5, y + 0.5, dx / num_max / 2.1, dy / num_max / 2.1, facecolor="black", width=0.1)
                        policy_plot.add_patch(p)

        if show_grid:
            # Add agent
            image_coordinates = position_to_image_coordinates(self.agent_position, self.num_cols)
            policy_plot.imshow(self.agent_image, extent=image_coordinates)
            if show_state_labels:
                for state in range(self.num_states):
                    if state not in self.terminal_states:
                        x, y = one_d_to_two_d(state, self.num_cols)
                        policy_plot.text(x + 0.5, y + 0.5, state)

        if show_learning_curve:
            # Show Learning curve plot
            rewards_to_plot = reward_per_episode[:episode]
            episodes_so_far = np.arange(0, episode)
            lc_plot.plot(episodes_so_far, rewards_to_plot)
            lc_plot.set_xlabel('Episode')
            lc_plot.set_ylabel('Reward')

        plt.show()
