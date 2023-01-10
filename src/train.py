import torch
import numpy as np
from src.agents.ddpg import DDPG
from src.agents.ppo import PPO
from src.envs.robot_arm_env import RobotArmEnv

# Configuration
STATE_DIM = 4  # Joint angles (2) + end-effector pos (2)
ACTION_DIM = 2 # Joint angle changes
MAX_ACTION = np.pi / 10

# DDPG Parameters
DDPG_LR_ACTOR = 1e-4
DDPG_LR_CRITIC = 1e-3
DDPG_GAMMA = 0.99
DDPG_TAU = 0.005
DDPG_BUFFER_CAPACITY = 100000
DDPG_BATCH_SIZE = 64

# PPO Parameters
PPO_LR_ACTOR = 3e-4
PPO_LR_CRITIC = 1e-3
PPO_GAMMA = 0.99
PPO_K_EPOCHS = 80
PPO_EPS_CLIP = 0.2
PPO_ACTION_STD_INIT = 0.6

MAX_EPISODES = 200
MAX_STEPS_PER_EPISODE = 200

def train_ddpg():
    env = RobotArmEnv()
    agent = DDPG(STATE_DIM, ACTION_DIM, MAX_ACTION, DDPG_LR_ACTOR, DDPG_LR_CRITIC, DDPG_GAMMA, DDPG_TAU, DDPG_BUFFER_CAPACITY, DDPG_BATCH_SIZE)

    print("
--- Training DDPG Agent ---")
    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            # Add some noise for exploration
            action = (action + np.random.normal(0, 0.1, size=ACTION_DIM)).clip(-MAX_ACTION, MAX_ACTION)
            
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.train()
            if done:
                break
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")

def train_ppo():
    env = RobotArmEnv()
    ppo_agent = PPO(STATE_DIM, ACTION_DIM, True, PPO_ACTION_STD_INIT, PPO_LR_ACTOR, PPO_LR_CRITIC, PPO_GAMMA, PPO_K_EPOCHS, PPO_EPS_CLIP)

    # Dummy memory for PPO
    class PPOMemory:
        def __init__(self):
            self.actions = []
            self.states = []
            self.logprobs = []
            self.rewards = []
            self.is_terminals = []

        def clear_memory(self):
            del self.actions[:]
            del self.states[:]
            del self.logprobs[:]
            del self.rewards[:]
            del self.is_terminals[:]

    memory = PPOMemory()

    print("
--- Training PPO Agent ---")
    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        for step in range(MAX_STEPS_PER_EPISODE):
            action, logprob, state_val = ppo_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Storing data in PPO memory
            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.FloatTensor(action))
            memory.logprobs.append(torch.FloatTensor(logprob))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break
        
        # Update PPO policy after each episode
        ppo_agent.update(memory)
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")


if __name__ == "__main__":
    train_ddpg()
    train_ppo()
