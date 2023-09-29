import os
import glob
import gymnasium as gym
import gymnasium.vector 
from collections import deque
from omegaconf import DictConfig
import hydra
import logging
import time
import numpy as np  
import threading
import wandb
import wandb.plot as wplot

from datetime import datetime
from pathlib import Path

from memory import VectorizedReplayBuffer
from dqn_agent import DQNAgent, DoubleDQNAgent
from utils import set_seed, create_annotated_video
from utils import get_agent, add_action_dist



def evaluate_agent(agent, env_name, logger, episode, num_episodes=10):
    """ """
    env = gym.make(env_name)
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset(seed=episode)
        
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state, evaluate_agent=True)
            state, reward, done, tranc, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    avg_reward = sum(rewards) / num_episodes
    min_reward = min(rewards)
    max_reward = max(rewards)
    logger.info(f"Evaluation at episode {episode} over {num_episodes} episodes: {avg_reward:.2f} min: {min_reward:.2f} max: {max_reward:.2f}")

    
@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    day = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H-%M-%S")
    # import pdb; pdb.set_trace()
    experiment_path = os.path.join(os.path.expanduser("~"), "experiments",  day, current_time)
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    # intialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    log_file_path = os.path.join(experiment_path, "logs")
    Path(log_file_path).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path + "/log.log")
    logger.addHandler(file_handler)
    env_name = cfg.env.name
    num_processes = cfg.env.num_processes
    total_episodes = cfg.env.total_episodes
    buffer_size = cfg.agent.buffer_size
    cfg.agent.updates = cfg.env.num_processes
    # Initialize Lunarlander environment and policy
    env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_processes)])


    set_seed(cfg.env.seed)
    env.action_space.seed(cfg.env.seed)
    cfg.agent.action_dim = int(env.action_space[0].n)
    cfg.agent.state_dim = env.observation_space.shape[1]
    # Initialize agent
    agent, run_name = get_agent(cfg, day, current_time)
    
    

    # Initialize replay memory
    memory = VectorizedReplayBuffer(capacity=buffer_size, state_dim=cfg.agent.state_dim, logger=logger)
     # Create a unique run name based on hyperparameters and current time
    

    # Initialize wandb with a run name and a group name
    if  cfg.track:
        job_type = f"batch_size_{cfg.agent.batch_size}_buffer_size_{buffer_size}_updates_{cfg.agent.updates}"
        wandb_run = wandb.init(project="lunar-lander", 
                            config=cfg, 
                            name=run_name, 
                            job_type= job_type,
        )
        
        # Create an artifact for the code
        code_artifact = wandb.Artifact(
            name="code",
            type="code",
            description="Codebase for this run"
        )
        # Get the original working directory
        original_cwd = hydra.utils.get_original_cwd()
        
        # Add all Python and YAML files to the artifact from the original directory
        for ext in ["py", "yaml"]:
            for filename in glob.glob(os.path.join(original_cwd, f"**/*.{ext}"), recursive=True):
                # We only add the file if it's not inside a hidden directory (those that start with ".")
                if "/." not in filename:
                    # import pdb; pdb.set_trace()
                    relative_path = os.path.relpath(filename, original_cwd)  # Get relative path to preserve folder structure
                    code_artifact.add_file(filename, name=relative_path)

        # Log the artifact
        wandb_run.log_artifact(code_artifact)
    
    start = time.time()
    mean_loss = deque(maxlen=100)
    # start episode
    # mean reward of 100 episodes
    create_annotated_video(agent=agent, env_name=env_name, path=experiment_path, sec_passed=0, num_episodes=1)
    evaluate_agent(agent, env_name, logger, 0)
    
    action_dist = [1 for _ in range(cfg.agent.action_dim)]
    mean_reward = deque(maxlen=100)
    for episode in range(1, total_episodes + 1):
        # reset environment
        states, _  = env.reset(seed=cfg.env.seed)
        # import pdb; pdb.set_trace()
        total_reward = [0 for _ in range(num_processes)]
        steps = 0
        while True:
            steps += 1
            # Epsilon-greedy action selection
            if agent.epsilon < np.random.rand():
                actions = agent.select_action(states)
                
                
            else:
                actions = env.action_space.sample()
             
            
            action_dist = add_action_dist(actions=actions, action_dist=action_dist)
            # normalize action distribution
            action_dist_normalized = [action / sum(action_dist) for action in action_dist]
            next_states, rewards, dones, truncs, _ = env.step(actions)
            if episode >= cfg.agent.start_learning_episode:
                # Update agent
                batch = memory.sample(cfg.agent.batch_size)
                loss = agent.update(batch)
                mean_loss.append(loss)
            # Store experience in replay memory
            memory.push(states, actions, rewards, next_states, dones)
            states = next_states
            total_reward[0] += rewards[0]
            # end episode if done is true for any of the processes
            if np.any(dones) or steps > 200:
                mean_reward.append(total_reward[0])
                break
                # import pdb; pdb.set_trace()
        if episode % 20 == 0:
            passed = time.time() - start
            if cfg.track:
                wandb.log({"mean_reward": np.mean(mean_reward), "epsilon": agent.epsilon, "loss": loss, "mean loss": np.mean(mean_loss), "memory_size": memory.size})
                for idx, a in  enumerate(action_dist_normalized):
                    wandb.log({f"action_dist_{idx}": a})
                
            logger.info(f"Episode: {episode}, mean Reward {np.mean(mean_reward):.2f}  Reward: {total_reward[0]:.2f}, Epsilon: {agent.epsilon:.2f}, memory size {memory.size} Time: {time.time() - start:.2f}")
            create_annotated_video(agent=agent, env_name=env_name, path=experiment_path, sec_passed=passed, num_episodes=episode)
            eval_thread = threading.Thread(target=evaluate_agent, args=(agent, cfg.env.name, logger, episode))
            eval_thread.start()
            





if __name__ == '__main__':
    main()