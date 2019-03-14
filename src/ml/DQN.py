from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add


class DQNAgent(object):
    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self, game, snake, apple):

        state = [
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
            ),  # danger straight
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
            ),  # danger right
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
            ),  # danger left
            snake.x_change == -20,  # move left
            snake.x_change == 20,  # move right
            snake.y_change == -20,  # move up
            snake.y_change == 20,  # move down
            apple.x_apple < snake.x,  # apple left
            apple.x_apple > snake.x,  # apple right
            apple.y_apple < snake.y,  # apple up
            apple.y_apple > snake.y,  # apple down
        ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def set_reward(self, snake, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if snake.eaten:
            self.reward = 10
        return self.reward

    def network(self, weights=None):
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
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(
                self.model.predict(next_state.reshape((1, 11)))[0]
            )
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
