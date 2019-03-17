import os
from random import randint

import numpy as np
import pygame

from snake_machine_learning.game.game import eat, update_screen
from snake_machine_learning.game.game_config import game_config

# Get the path of the current file
path = os.path.dirname(os.path.abspath(__file__))


class Game:
    """
    A class used to represent a game

    ...

    Attributes
    ----------
    game_width: int
        Width of tha game window
    game_height: int
        Height of the game window
    game_display: PyGame display
        Surface for the pygame
    bg: image
        Image fpr the background of the game
    crash: boolean
        Has the snake crashed
    score: int
        Score of the game
    snake: Snake object
        The snake of the game
    apple: Apple object
        Apple in the game
    """

    def __init__(self):
        pygame.font.init()
        pygame.display.set_caption(game_config["game"]["caption"])
        self.game_width = game_config["game"]["width"]
        self.game_height = game_config["game"]["height"]
        self.game_display = pygame.display.set_mode(
            (game_config["game"]["width"], game_config["game"]["height"] + 60)
        )
        self.bg = pygame.image.load(path + "/img/background.png").convert_alpha()
        self.crash = False
        self.snake = Snake(self)
        self.apple = Apple()
        self.score = 0


class Snake(object):
    """
    A class used to represent a snake

    ...

    Attributes
    ----------
    x : int
        X position of the snake
    y : int
        Y position of the snake
    position: int[]
        2d array of the snake's position
    apple: int
        How many apples long the snake is
    eaten: boolean
        Has the snake eaten
    image: image
        image for the body of the game
    x_change: Change in the x direction
    y_change: Change in the y direction
    """

    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % game_config["snake"]["width"]
        self.y = y - y % game_config["snake"]["height"]
        self.position = []
        self.position.append([self.x, self.y])
        self.apple = 1
        self.eaten = False
        self.image = pygame.image.load(path + "/img/snake_body.png").convert_alpha()
        self.x_change = game_config["snake"]["width"]
        self.y_change = 0

    def update_position(self, x, y):
        """Update the snake position for each move

        :param x: New x position
        :param y: New y position
        """
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.apple > 1:
                for i in range(0, self.apple - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, apple):
        """Perform a given move

        :param move: Move to be performed
        :param x: Snake x position
        :param y: Snake y position
        :param game: Game object
        :param apple: Apple object
        """

        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.apple = self.apple + 1

        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif (
            np.array_equal(move, [0, 1, 0]) and self.y_change == 0
        ):  # Right - going horizontal
            move_array = [0, self.x_change]
        elif (
            np.array_equal(move, [0, 1, 0]) and self.x_change == 0
        ):  # Right - going vertical
            move_array = [-self.y_change, 0]
        elif (
            np.array_equal(move, [0, 0, 1]) and self.y_change == 0
        ):  # Left - going horizontal
            move_array = [0, -self.x_change]
        elif (
            np.array_equal(move, [0, 0, 1]) and self.x_change == 0
        ):  # Left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if (
            self.x < 20
            or self.x > game.game_width - 40
            or self.y < 20
            or self.y > game.game_height - 40
            or [self.x, self.y] in self.position
        ):
            game.crash = True
        eat(self, apple, game)

        self.update_position(self.x, self.y)

    def display_snake(self, x, y, apple, game):
        """Display snake on the game surface

        :param x: Snake x position
        :param y: Snake y position
        :param apple: Apple object
        :param game: Game object
        """
        self.position[-1][0] = x
        self.position[-1][1] = y

        if not game.crash:
            for i in range(apple):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.game_display.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


class Apple(object):
    """
    A class used to represent a snake

    ...

    Attributes
    ----------
    x_apple : int
        X position of the apple
    y_apple : int
        Y position of the apple
    image: image
        image for the apple
    """

    def __init__(self):
        self.x_apple = 240
        self.y_apple = 200
        self.image = pygame.image.load(path + "/img/apple.png").convert_alpha()

    def apple_coord(self, game, snake):
        """Create coordinates for the apple

        :param game: Game object
        :param snake: Snake object
        :return: Coordinates for the apple
        """
        x_rand = randint(game_config["apple"]["width"], game.game_width - 40)
        self.x_apple = x_rand - x_rand % game_config["apple"]["width"]
        y_rand = randint(game_config["apple"]["height"], game.game_height - 40)
        self.y_apple = y_rand - y_rand % game_config["apple"]["height"]
        if [self.x_apple, self.y_apple] not in snake.position:
            return self.x_apple, self.y_apple
        else:
            self.apple_coord(game, snake)

    def display_apple(self, x, y, game):
        """Display apple on the game surface

        :param x: Apple x position
        :param y: Apple y position
        :param game: Game object
        """
        game.game_display.blit(self.image, (x, y))
        update_screen()
