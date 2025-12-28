<table>
  <tr>
    <td valign="top">
      <img src="./assets/icon.png" alt="Project Icon" width="150"/>
    </td>
    <td valign="top">
      <h1>🐍 Reinforcement Learning Snake Project</h1>
      <p>Welcome to the RL Snake project! This guide will walk you through setting up the development environment so you can start coding.</p>
    </td>
  </tr>
</table>

---

## 🚀 Getting Started: Environment Setup

We use `conda` to manage our project's environment and dependencies. This ensures everyone has the exact same setup and avoids "it works on my machine" issues.

### 1.  Create the Conda Environment 🛠️

Navigate to the project's root directory (where the `environment.yml` file is located) in your terminal and run the following command:

```bash
conda env create -f environment.yml
```

This command will read the `environment.yml` file and install all the necessary packages (like PyTorch, Pygame, etc.) into a new environment named `rl_snake_env`.

### 2. Activate the Environment ✅

Once the environment is created, you need to activate it every time you work on the project. Run this command:

```bash
conda activate rl_snake_env
```

Your terminal prompt should now show `(rl_snake_env)` at the beginning, indicating that the environment is active.

---

## 💻 Current Progress

We have successfully implemented the core game environment and multiple RL agents!  Here's a quick summary of what's done:

*   **Game Environment (`game.py`)** 🕹️: The main `SnakeGame` class is built using Pygame. It handles the game window, drawing, and core logic.
*   **Snake & Food Mechanics** 🍎: The snake can move, eat food, and grow longer. Food spawns at random locations.
*   **RL-Ready API** 🤖: The environment exposes `step(action)` and `reset()` methods, making it ready for an RL agent to interact with.  It returns `(reward, done, score)`.
*   **Collision Detection** 💥: The game correctly detects collisions with walls and the snake's own body, ending the episode.
*   **Manual Testing Mode** 👨‍💻: You can run `game.py` directly to play the game with your keyboard, which is great for testing!

### 🧠 Implemented RL Algorithms

*   **Double Q-Learning Agent** 📊:
    - Tabular approach with dual Q-tables to reduce overestimation bias
    - ✅ Performs well on **small grids** (320×240, 400×400)
    - ⚠️ Struggles with **large grids** due to exponential state space growth
    - Best for standard game sizes with limited complexity

*   **Deep Q-Network (DQN) Agent** 🚀:
    - Neural network-based approach using PyTorch (11 → 256 → 3 architecture)
    - ✅ Excellent performance on **large grids** (600×600, 800×800, 1000×1000)
    - ✅ Linear learning progress with consistent improvement over training
    - GPU-accelerated training with experience replay
    - Complete game replay system with configurable frame recording
    - Best for scalable, high-performance training across variable map sizes

*   **Dueling DQN Agent** 🎯:
    - Enhanced DQN with separate Value and Advantage streams
    - Improved state value estimation through dual-stream architecture
    - Better performance on complex decision-making scenarios

*   **Proximal Policy Optimization (PPO) Agent** 🌟:
    - Policy-based approach with Actor-Critic architecture
    - ✅ Stable learning through clipped objective function
    - ✅ Direct policy optimization without Q-value overestimation
    - On-policy learning with entropy regularization
    - Excellent exploration through stochastic policy sampling
    - Best for environments requiring stable, consistent policy improvement

---

## 🎮 Quick Start

### Train Double Q-Learning Agent (Small Grids)
```bash
python Double_QLearning/train.py
```

### Train DQN Agent (Large Grids)
```bash
python Deep_QLearning/train.py
```

### Train Dueling DQN Agent
```bash
python "Dueling DQN/train.py"
```

### Train PPO Agent (Policy-Based)
```bash
python PPO/train.py
```

### Watch Replays
```bash
python replay_best_game.py
```

---

## 📊 Performance Comparison

| Algorithm | Type | Grid Size | Training Speed | Scalability | Best Use Case |
|-----------|------|-----------|----------------|-------------|---------------|
| Double Q-Learning | Value-based | 320×240 | ⚡ Fast (50-100 games/s) | ⚠️ Limited | Small, simple environments |
| DQN | Value-based | 600×600+ | ⚡ Fast (100+ games/s) | ✅ Excellent | Large, complex environments |
| Dueling DQN | Value-based | 600×600+ | ⚡ Fast (100+ games/s) | ✅ Excellent | Complex decision-making |
| PPO | Policy-based | 600×600+ | ⚡ Medium (50-80 games/s) | ✅ Excellent | Stable policy learning |

---

## 🔧 Optimization Tips

- **DQN Memory Optimization**: Uncomment the score threshold in `DQNAgent.record_frame()` to reduce memory usage during training
- **GPU Acceleration**: Ensure CUDA-compatible GPU for 2-3x faster DQN training
- **Curriculum Learning**: Start with small grids, progressively increase size for better convergence

---