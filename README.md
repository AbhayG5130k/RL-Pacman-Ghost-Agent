# NeuroMaze: Reinforcement Learning Ghost Agent

A Reinforcement Learning-based Pac-Man ghost agent built using Python and Pygame.

This project implements a Q-Learning agent that learns how to chase and capture a randomly moving Pac-Man player inside a maze environment. The agent improves its movement strategy over multiple training episodes using reward-based learning.

---

## Features

- Q-Learning implementation from scratch
- Epsilon-greedy exploration strategy
- Reward shaping for efficient learning
- Real-time training and gameplay visualization
- Custom maze environment using Pygame
- Autonomous ghost AI behavior
- Training and inference modes

---

## Technologies Used

- Python
- Pygame
- NumPy
- Reinforcement Learning (Q-Learning)

---

## Reinforcement Learning Concepts Used

### State Representation
The environment state consists of:
- Ghost position `(x, y)`
- Player position `(x, y)`

### Actions
The ghost agent can:
- Move Up
- Move Down
- Move Left
- Move Right

### Rewards
The agent receives:
- Positive rewards for moving closer to the player
- Negative rewards for moving away
- Large positive reward for catching the player
- Penalty for invalid wall collisions

### Learning Strategy
The agent uses:
- Q-Table learning
- Bellman Equation updates
- Epsilon-greedy exploration

---

## Training Process

The ghost agent trains over multiple episodes and gradually improves its chasing behavior.

During training:
- The Pac-Man player moves randomly
- The ghost learns optimal movement paths
- Exploration decreases over time using epsilon decay

---

## Results

After training:
- The ghost learns to efficiently pursue the player
- Invalid wall movements reduce significantly
- The agent develops better path selection strategies

---

## Future Improvements

- Deep Q-Networks (DQN)
- Multiple ghost agents
- Smarter Pac-Man behavior
- Pathfinding optimization
- Dynamic reward systems
- Save/load trained models

---

## How to Run

### Install dependencies

```bash
pip install pygame numpy
