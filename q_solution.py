import numpy as np
import time


class AgentQ:
    def __init__(self, env, policy="epsilon_greedy", epsilon=0.05, alpha=0.1, gamma=1):
        self.env = env
        self.q_table = np.zeros(shape=(self.env.num_states, self.env.num_actions))
        self.policy = policy
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self):
        if self.policy == "epsilon_greedy" and np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.env.num_actions)
        else:
            state = self.env.agent_position
            q_values_of_state = self.q_table[state, :]
            # Choose randomly AMONG maximum Q-values
            max_q_value = np.max(q_values_of_state)
            maximum_q_values = np.nonzero(q_values_of_state == max_q_value)[0]
            action = np.random.choice(maximum_q_values)
        return action

    def learn(self, old_state, reward, new_state, action):
        max_q_value_in_new_state = np.max(self.q_table[new_state, :])
        current_q_value = self.q_table[old_state, action]
        self.q_table[old_state, action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)


def q_learning(env, agent, num_episodes=500, max_steps_per_episode=1000, learn=True, seconds_between_each_step=0,
               show_grid=False, show_policy=False, show_q_values=False, show_softmax=False, show_learning_curve=False,
               fig_size=6):
    reward_per_episode = np.zeros(num_episodes)
    for episode in range(0, num_episodes):
        cumulative_reward = 0
        step = 0
        game_over = False
        while step < max_steps_per_episode and not game_over:
            time.sleep(seconds_between_each_step)
            if show_grid or show_learning_curve:
                env.visualize(show_grid=show_grid, show_policy=show_policy,
                              show_learning_curve=show_learning_curve,
                              show_q_values=show_q_values, clear_the_output=True,
                              episode=episode, reward_per_episode=reward_per_episode,
                              agent_q_table=agent.q_table, fig_size=fig_size)

            old_state = env.agent_position
            action = agent.choose_action()
            reward, new_state = env.make_step(action)
            if learn:
                agent.learn(old_state, reward, new_state, action)
            cumulative_reward += reward
            step += 1

            # Check whether agent is at terminal state. If yes: end episode; reset agent.
            if env.is_terminal_state():
                time.sleep(seconds_between_each_step)
                if show_grid or show_learning_curve:
                    env.visualize(show_grid=show_grid, show_policy=show_policy,
                                  show_learning_curve=show_learning_curve,
                                  show_q_values=show_q_values, clear_the_output=True,
                                  episode=episode, reward_per_episode=reward_per_episode,
                                  agent_q_table=agent.q_table, fig_size=fig_size)
                env.reset()
                game_over = True

        reward_per_episode[episode] = cumulative_reward
    return reward_per_episode



# ##
# ## Leightweight solutions
# ##
# class AgentQ:
#     def __init__(self, num_states, num_actions, epsilon=0.05, alpha=0.1):
#         self.num_actions = num_actions
#         self.q_table = np.zeros(shape=(num_states, self.num_actions))
#         self.epsilon = epsilon
#         self.alpha = alpha
#
#     def choose_action(self, current_state):
#         if np.random.uniform(0, 1) < self.epsilon:
#             action = np.random.randint(0, self.num_actions)
#         else:
#             q_values_of_state = self.q_table[current_state, :]
#             action = np.argmax(q_values_of_state)
# #             # Choose randomly AMONG maximum Q-values
# #             max_q_value = np.max(q_values_of_state)
# #             maximum_q_values = np.nonzero(q_values_of_state == max_q_value)[0]
# #             action = np.random.choice(maximum_q_values)
#         return action
#
#     def learn(self, old_state, reward, new_state, action):
#         max_q_value_in_new_state = np.max(self.q_table[new_state, :])
#         current_q_value = self.q_table[old_state, action]
#         self.q_table[old_state, action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + max_q_value_in_new_state)
#
#
# def q_learning(env, agent, visualize=True, num_episodes=10, sleep_between_each_step=0):
#     for episode in range(num_episodes):
#         env.reset()
#         game_over = False
#         while not game_over:
#             old_state = env.agent_position
#             action = agent.choose_action(old_state)
#             reward, new_state = env.make_step(action)
#             agent.learn(old_state, reward, new_state, action)
#             env.visualize()
#             # Check whether agent is at terminal state. If yes: end episode; reset agent.
#             if env.is_terminal_state():
#                 game_over = True

