from random import randint
import pygame

from game.game_config import Config
import numpy as np

from game.game import update_screen, eat


class Game:
    def __init__(self):
        pygame.font.init()
        pygame.display.set_caption(Config["game"]["caption"])
        self.game_width = Config["game"]["width"]
        self.game_height = Config["game"]["height"]
        self.game_display = pygame.display.set_mode(
            (Config["game"]["width"], Config["game"]["height"])
        )
        self.bg = pygame.image.load("game/img/background.png").convert_alpha()
        self.crash = False
        self.snake = Snake(self)
        self.apple = Apple()
        self.score = 0


class Snake(object):
    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % Config["snake"]["width"]
        self.y = y - y % Config["snake"]["height"]
        self.position = []
        self.position.append([self.x, self.y])
        self.apple = 1
        self.eaten = False
        self.image = pygame.image.load("game/img/snake_body.png").convert_alpha()
        self.x_change = Config["snake"]["width"]
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.apple > 1:
                for i in range(0, self.apple - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, apple):
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
    def __init__(self):
        self.x_apple = 240
        self.y_apple = 200
        self.image = pygame.image.load("game/img/apple.png").convert_alpha()

    def apple_coord(self, game, player):
        x_rand = randint(Config["apple"]["width"], game.game_width - 40)
        self.x_apple = x_rand - x_rand % Config["apple"]["width"]
        y_rand = randint(Config["apple"]["height"], game.game_height - 40)
        self.y_apple = y_rand - y_rand % Config["apple"]["height"]
        if [self.x_apple, self.y_apple] not in player.position:
            return self.x_apple, self.y_apple
        else:
            self.apple_coord(game, player)

    def display_apple(self, x, y, game):
        game.game_display.blit(self.image, (x, y))
        update_screen()
