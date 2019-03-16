from random import randint

import numpy as np
import pygame
from keras.utils import to_categorical

from snake_machine_learning.game.game import display, get_record, initialize_game
from snake_machine_learning.game.game_classes import Game
from snake_machine_learning.game.game_config import game_config
from snake_machine_learning.ml.DQN import DQNAgent


def main():
    pygame.init()
    agent = DQNAgent()
    counter_games = 0
    record = 0

    while counter_games < 150:
        # Initialize classes
        game = Game()
        snake1 = game.snake
        apple1 = game.apple

        # Perform first move
        initialize_game(snake1, game, apple1, agent)
        display(snake1, apple1, game, record)

        while not game.crash:
            # Agent.epsilon is set to give randomness to actions
            agent.epsilon = 80 - counter_games

            # Get old state
            state_old = agent.get_state(game, snake1, apple1)

            # Perform random actions based on agent.epsilon, or choose the action
            if randint(0, 200) < agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                # Predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            # Perform new move and get new state
            snake1.do_move(final_move, snake1.x, snake1.y, game, apple1)
            state_new = agent.get_state(game, snake1, apple1)

            # Set reward for the new state
            reward = agent.set_reward(snake1, game.crash)

            # Train short memory base on the new action and state
            agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)

            # Store the new data into a long term memory
            agent.remember(state_old, final_move, reward, state_new, game.crash)
            record = get_record(game.score, record)
            display(snake1, apple1, game, record)
            pygame.time.wait(game_config["speed"])

        agent.replay_new(agent.memory)
        counter_games += 1
        print("Game", counter_games, "      Score:", game.score)


if __name__ == "__main__":
    main()
