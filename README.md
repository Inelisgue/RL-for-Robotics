# RL-for-Robotics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=flat-square&logo=numpy)](https://numpy.org/)

A Python project exploring Reinforcement Learning algorithms for controlling robotic systems. This repository includes simulations of robotic arms learning to perform manipulation tasks using DDPG and PPO algorithms.

## ✨ Features

-   **DDPG (Deep Deterministic Policy Gradient)**: Implementation of a continuous control RL algorithm.
-   **PPO (Proximal Policy Optimization)**: Implementation of a policy gradient RL algorithm.
-   **Robot Arm Environment**: A simplified simulation environment for robotic arm control.
-   **Learning Visualization**: Tools to observe the learning progress of the agents.

## 🚀 Getting Started

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Inelisgue/RL-for-Robotics.git
    cd RL-for-Robotics
    ```
2.  Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To train both DDPG and PPO agents on the simulated robot arm environment:

```bash
python src/train.py
```

This script will run a series of episodes for each algorithm and print the episodic rewards.

## 📚 Project Structure

```
RL-for-Robotics/
├── src/
│   ├── agents/             # Implementations of DDPG and PPO algorithms
│   ├── envs/               # Robotic arm simulation environment
│   └── train.py            # Main script for training RL agents
├── experiments/            # Scripts for running specific experiments
├── models/                 # Directory for saving trained models
├── README.md               # Project overview and documentation
├── requirements.txt        # Python dependencies
└── .gitignore              # Git ignore file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
