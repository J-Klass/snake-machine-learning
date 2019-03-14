import pygame


def eat(snake, apple, game):
    if snake.x == apple.x_apple and snake.y == apple.y_apple:
        apple.apple_coord(game, snake)
        snake.eaten = True
        game.score = game.score + 1


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def display_ui(game, score, record):
    myfont = pygame.font.SysFont("Segoe UI", 20)
    myfont_bold = pygame.font.SysFont("Segoe UI", 20, True)
    text_score = myfont.render("SCORE: ", True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render("HIGHEST SCORE: ", True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (0, 0))


def display(snake, apple, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    snake.display_snake(snake.position[-1][0], snake.position[-1][1], snake.apple, game)
    apple.display_apple(apple.x_apple, apple.y_apple, game)


def update_screen():
    pygame.display.update()


def initialize_game(snake, game, apple, agent):
    state_init1 = agent.get_state(game, snake, apple)
    action = [1, 0, 0]
    snake.do_move(action, snake.x, snake.y, game, apple, agent)
    state_init2 = agent.get_state(game, snake, apple)
    reward1 = agent.set_reward(snake, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory)
