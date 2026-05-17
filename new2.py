import pygame
import sys
import random
import numpy as np

TRAINING_MODE = True

EPISODES = 5000

CELL_SIZE = 24


maze = [
    "############################",
    "#............##............#",
    "#.######.###.##.###.######.#",
    "#.#....#..............#...#.#",
    "#.######.##.####.##.######.#",
    "#..........................#",
    "####.#####.######.#####.####",
    "#............##............#",
    "#.######.###.##.###.######.#",
    "#......#..............#....#",
    "######.#.######.####.#.#####",
    "#............P.............#",
    "############################"
]

ROWS = len(maze)
COLS = len(maze[0])

WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE


if not TRAINING_MODE:

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("Pac-Man RL")

    clock = pygame.time.Clock()


BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)


for y in range(ROWS):
    for x in range(COLS):

        if maze[y][x] == "P":

            PLAYER_START_X = x
            PLAYER_START_Y = y



player_x = PLAYER_START_X
player_y = PLAYER_START_Y

enemy = {
    "x": 13,
    "y": 5
}

# RL SETTINGS


ACTIONS = {
    0: (0, -1),   # UP
    1: (0, 1),    # DOWN
    2: (-1, 0),   # LEFT
    3: (1, 0)     # RIGHT
}

NUM_ACTIONS = 4

q_table = {}

alpha = 0.1
gamma = 0.9

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# ==========================================
# FUNCTIONS
# ==========================================

def can_move(x, y):

    if x < 0 or x >= COLS or y < 0 or y >= ROWS:
        return False

    return maze[y][x] != "#"



def draw_maze():

    for row in range(ROWS):
        for col in range(COLS):

            tile = maze[row][col]

            x = col * CELL_SIZE
            y = row * CELL_SIZE

            if tile == "#":

                pygame.draw.rect(
                    screen,
                    BLUE,
                    (x, y, CELL_SIZE, CELL_SIZE),
                    border_radius=4
                )

            elif tile == ".":

                pygame.draw.circle(
                    screen,
                    WHITE,
                    (
                        x + CELL_SIZE // 2,
                        y + CELL_SIZE // 2
                    ),
                    3
                )


def draw_player():

    pygame.draw.circle(
        screen,
        YELLOW,
        (
            player_x * CELL_SIZE + CELL_SIZE // 2,
            player_y * CELL_SIZE + CELL_SIZE // 2
        ),
        CELL_SIZE // 2 - 2
    )


def draw_enemy():

    pygame.draw.circle(
        screen,
        RED,
        (
            enemy["x"] * CELL_SIZE + CELL_SIZE // 2,
            enemy["y"] * CELL_SIZE + CELL_SIZE // 2
        ),
        CELL_SIZE // 2 - 2
    )
# RL ENVIRONMENT


def get_state():

    return (
        enemy["x"],
        enemy["y"],
        player_x,
        player_y
    )


def reset():

    global player_x, player_y

    player_x = PLAYER_START_X
    player_y = PLAYER_START_Y

    enemy["x"] = 13
    enemy["y"] = 5

    return get_state()


def move_player_random():

    global player_x, player_y

    directions = [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0)
    ]

    random.shuffle(directions)

    for dx, dy in directions:

        next_x = player_x + dx
        next_y = player_y + dy

        if can_move(next_x, next_y):

            player_x = next_x
            player_y = next_y

            break


def step(action):

    dx, dy = ACTIONS[action]

    current_distance = (
        abs(enemy["x"] - player_x) +
        abs(enemy["y"] - player_y)
    )

    next_x = enemy["x"] + dx
    next_y = enemy["y"] + dy

    reward = -1
    done = False

    if not can_move(next_x, next_y):

        reward = -10

    else:

        enemy["x"] = next_x
        enemy["y"] = next_y


    new_distance = (
        abs(enemy["x"] - player_x) +
        abs(enemy["y"] - player_y)
    )

    if new_distance < current_distance:
        reward += 2

    
    elif new_distance > current_distance:
        reward -= 2

    
    if enemy["x"] == player_x and enemy["y"] == player_y:

        reward = 100
        done = True

    next_state = get_state()

    return next_state, reward, done

# Q-LEARNING


def initialize_state(state):

    if state not in q_table:

        q_table[state] = np.zeros(NUM_ACTIONS)


def choose_action(state):

    initialize_state(state)

    # EXPLORE
    if random.random() < epsilon:

        return random.randint(0, NUM_ACTIONS - 1)

    # EXPLOIT
    return np.argmax(q_table[state])


def update_q_table(state, action, reward, next_state):

    initialize_state(next_state)

    old_q = q_table[state][action]

    max_future_q = np.max(q_table[next_state])

    new_q = old_q + alpha * (
        reward +
        gamma * max_future_q -
        old_q
    )

    q_table[state][action] = new_q

# TRAINING LOOP


episode = 0

state = reset()

while True:

   

    if TRAINING_MODE:

        move_player_random()

        action = choose_action(state)

        next_state, reward, done = step(action)

        update_q_table(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

        if done:

            episode += 1

            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

            print(
                f"Episode: {episode} | "
                f"Epsilon: {epsilon:.3f}"
            )

            state = reset()

        if episode >= EPISODES:

            print("\nTRAINING FINISHED")
            print("Q-table size:", len(q_table))

            TRAINING_MODE = False

            pygame.init()

            screen = pygame.display.set_mode(
                (WIDTH, HEIGHT)
            )

            pygame.display.set_caption(
                "Trained RL Ghost"
            )

            clock = pygame.time.Clock()

            state = reset()



    else:

        screen.fill(BLACK)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()
                sys.exit()

        
        move_player_random()

        
        initialize_state(state)

        action = np.argmax(q_table[state])

        next_state, reward, done = step(action)

        state = next_state

        if done:

            print("PLAYER CAUGHT")

            pygame.time.delay(1000)

            state = reset()

        draw_maze()
        draw_player()
        draw_enemy()

        pygame.display.update()

        clock.tick(15)