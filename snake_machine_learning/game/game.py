import pygame

from snake_machine_learning.game.game_config import game_config


def eat(snake, apple, game):
    """Appends an apple to the snake body if an apple has been eaten

    :param snake: The snake object
    :param apple: The apple object
    :param game: The game object
    """

    if snake.x == apple.x_apple and snake.y == apple.y_apple:
        apple.apple_coord(game, snake)
        snake.eaten = True
        game.score = game.score + 1


def get_record(score, record):
    """Checks if the new score is a new highscore

    :param score: Score of the game
    :param record: Highscore so far
    :return: Score or highscore
    """
    if score >= record:
        return score
    else:
        return record


def display_ui(game, score, record):
    """Display all the game's UI,

    :param game: Game object
    :param score: Score of the game
    :param record: Highest score overall
    """
    myfont = pygame.font.SysFont("arial", 20)
    myfont_bold = pygame.font.SysFont("arial", 20, True)
    text_score = myfont.render("SCORE: ", True, game_config["colors"]["black"])
    text_score_number = myfont_bold.render(
        str(score), True, game_config["colors"]["black"]
    )
    text_highest = myfont.render("HIGHEST SCORE: ", True, game_config["colors"]["black"])
    text_highest_number = myfont_bold.render(
        str(record), True, game_config["colors"]["black"]
    )
    game.game_display.blit(text_score, (20, 520))
    game.game_display.blit(text_score_number, (80, 520))
    game.game_display.blit(text_highest, (120, 520))
    game.game_display.blit(text_highest_number, (250, 520))
    game.game_display.blit(game.bg, (0, 0))


def display(snake, apple, game, record):
    """Display all game components

    :param snake: Snake object
    :param apple: Apple object
    :param game: Game object
    :param record: Highscore of the games
    """
    game.game_display.fill(game_config["colors"]["white"])
    display_ui(game, game.score, record)
    snake.display_snake(snake.position[-1][0], snake.position[-1][1], snake.apple, game)
    apple.display_apple(apple.x_apple, apple.y_apple, game)


def update_screen():
    """Update PyGame display"""

    pygame.display.update()


def initialize_game(snake, game, apple, agent):
    """Initialzie a game with the snake moving in one direction

    :param snake: Snake object
    :param game: Game object
    :param apple: Apple object
    :param agent: Agent object
    """
    state_init1 = agent.get_state(game, snake, apple)
    action = [1, 0, 0]
    snake.do_move(action, snake.x, snake.y, game, apple)
    state_init2 = agent.get_state(game, snake, apple)
    reward1 = agent.set_reward(snake, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory)
