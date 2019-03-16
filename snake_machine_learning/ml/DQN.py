from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add

from snake_machine_learning.ml.ml_config import ml_config


class DQNAgent(object):
    def __init__(self):
        self.reward = 0
        self.gamma = ml_config["gamma"]
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = ml_config["learning_rate"]
        self.model = self.network()
        self.epsilon = ml_config["epsilon"]
        self.actual = []
        self.memory = []

    def get_state(self, game, snake, apple):
        """
        Create all the input features for the Agent. Current state of the game.
        """

        state = [
            # Is there danger straight ahead
            (
                snake.x_change == 20
                and snake.y_change == 0
                and (
                    (list(map(add, snake.position[-1], [20, 0])) in snake.position)
                    or snake.position[-1][0] + 20 >= (game.game_width - 20)
                )
            )
            or (
                snake.x_change == -20
                and snake.y_change == 0
                and (
                    (list(map(add, snake.position[-1], [-20, 0])) in snake.position)
                    or snake.position[-1][0] - 20 < 20
                )
            )
            or (
                snake.x_change == 0
                and snake.y_change == -20
                and (
                    (list(map(add, snake.position[-1], [0, -20])) in snake.position)
                    or snake.position[-1][-1] - 20 < 20
                )
            )
            or (
                snake.x_change == 0
                and snake.y_change == 20
                and (
                    (list(map(add, snake.position[-1], [0, 20])) in snake.position)
                    or snake.position[-1][-1] + 20 >= (game.game_height - 20)
                )
            ),
            # Is there danger to the right
            (
                snake.x_change == 0
                and snake.y_change == -20
                and (
                    (list(map(add, snake.position[-1], [20, 0])) in snake.position)
                    or snake.position[-1][0] + 20 > (game.game_width - 20)
                )
            )
            or (
                snake.x_change == 0
                and snake.y_change == 20
                and (
                    (list(map(add, snake.position[-1], [-20, 0])) in snake.position)
                    or snake.position[-1][0] - 20 < 20
                )
            )
            or (
                snake.x_change == -20
                and snake.y_change == 0
                and (
                    (list(map(add, snake.position[-1], [0, -20])) in snake.position)
                    or snake.position[-1][-1] - 20 < 20
                )
            )
            or (
                snake.x_change == 20
                and snake.y_change == 0
                and (
                    (list(map(add, snake.position[-1], [0, 20])) in snake.position)
                    or snake.position[-1][-1] + 20 >= (game.game_height - 20)
                )
            ),
            # Is there danger to the left
            (
                snake.x_change == 0
                and snake.y_change == 20
                and (
                    (list(map(add, snake.position[-1], [20, 0])) in snake.position)
                    or snake.position[-1][0] + 20 > (game.game_width - 20)
                )
            )
            or (
                snake.x_change == 0
                and snake.y_change == -20
                and (
                    (list(map(add, snake.position[-1], [-20, 0])) in snake.position)
                    or snake.position[-1][0] - 20 < 20
                )
            )
            or (
                snake.x_change == 20
                and snake.y_change == 0
                and (
                    (list(map(add, snake.position[-1], [0, -20])) in snake.position)
                    or snake.position[-1][-1] - 20 < 20
                )
            )
            or (
                snake.x_change == -20
                and snake.y_change == 0
                and (
                    (list(map(add, snake.position[-1], [0, 20])) in snake.position)
                    or snake.position[-1][-1] + 20 >= (game.game_height - 20)
                )
            ),
            # Moving left
            snake.x_change == -20,
            # Moving right
            snake.x_change == 20,
            # Moving up
            snake.y_change == -20,
            # Moving down
            snake.y_change == 20,
            # Is apple to the lef
            apple.x_apple < snake.x,
            # Ia apple to the right
            apple.x_apple > snake.x,
            # Is the apple up
            apple.y_apple < snake.y,
            # Is apple down
            apple.y_apple > snake.y,
        ]

        # Transform boolean variables to 0's and 1's
        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def set_reward(self, snake, crash):
        """
        Create positive and negativ reward for the agent
        """

        self.reward = 0
        # Negative reward for crashing
        if crash:
            self.reward = ml_config["rewards"]["negative"]
            return self.reward
        # Positive reward for eating
        if snake.eaten:
            self.reward = ml_config["rewards"]["positive"]
        return self.reward

    def network(self, weights=None):
        """
        Model with the eleven input dimensions and 3 output dimensions for the next move.
        """
        model = Sequential()
        model.add(Dense(output_dim=120, activation="relu", input_dim=11))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation="relu"))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation="relu"))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=3, activation="softmax"))
        opt = Adam(self.learning_rate)
        model.compile(loss="mse", optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        """
        Creating new agent for next game
        """
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(np.array([next_state]))[0]
                )
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Predict the next move
        """
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(
                self.model.predict(next_state.reshape((1, 11)))[0]
            )
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
