import torch
import torch.nn as nn
import  numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import os

from gymnasium.wrappers import PixelObservationWrapper
import gymnasium as gym


# set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False 




def annotate_frame(frames, q_values, obs, current_reward=0, sum_rewards=0, frame_number=0):

    annotation_height = 50
    annotation = np.zeros((annotation_height, obs.shape[1], 3), dtype=np.uint8)
    annotation[:, :] = [0, 0, 255]
    num_actions = len(q_values[0])
    bar_width = obs.shape[1] // num_actions
    stripe_width = 5  # Width of the black stripe between bars

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color
    font_thickness = 1
    # Normalize Q-values for visualization
    # 
    q_values_norm = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
    # import pdb; pdb.set_trace()
    text_y = int(annotation_height * 0.5)  
    max_pos = np.argmax(q_values)
    for i, q_value in enumerate(q_values_norm[0]):
        start_x = i * (bar_width + stripe_width)
        end_x = start_x + bar_width - stripe_width  # Leave space for the black stripe
        # import pdb; pdb.set_trace()
        bar_height = int(q_value * annotation_height)
        
        if i == max_pos:
            annotation[-bar_height:, start_x:end_x] = [0, 255, 0]
            font_color = (255, 0, 0) 
        else:
            # import pdb; pdb.set_trace()
            #annotation[-bar_height:, start_x:end_x] = [0, 0, 255]  # RGB for red
            bar_height = annotation_height - bar_height
            annotation[:bar_height, start_x:end_x] = [0, 0, 0]  # RGB for red
            font_color = (255, 255, 255) 
        # Add the actual Q-value as text in the middle of the bar
        text = f"{q_values[0][i]:.3f}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = start_x + (bar_width - text_size[0]) // 2
        # print(f" text_x: {text_x} text_y: {text_y} bar height: {bar_height} text: {text} ")
        cv2.putText(annotation, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
    
    reward_text = f"Reward: {current_reward:.2f}, of {sum_rewards:.2f}"
    text_size = cv2.getTextSize(reward_text, font, font_scale, font_thickness)[0]
    text_x = obs.shape[1] - text_size[0] - 10  # 10 pixels padding from the right edge
    text_y = 20  # 20 pixels from the top
    cv2.putText(obs, reward_text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)  # Red color for rewards

    # Add frame number
    frame_text = f"Frame: {frame_number}"
    text_x = 10  # 10 pixels padding from the left edge
    text_y = 20  # 20 pixels from the top
    cv2.putText(obs, frame_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)  # White color for frame number



    # Combine the original frame and the annotation
    annotated_frame = np.vstack((obs, annotation))
    
    frames.append(annotated_frame)
    return frames

def create_annotated_video(agent, env_name, path, num_episodes=1):
    """ """
    # import pdb; pdb.set_trace()
    env = PixelObservationWrapper(gym.make(env_name, render_mode="rgb_array"),  pixels_only=False)
    for episode in range(num_episodes):
        state_dict, _ = env.reset()
        state = state_dict['state']
        obs = state_dict['pixels']
        frames = []
        steps = 0   
        episode_reward = 0
        while True:
            # Render the environment
            steps += 1
            
            # Get the Q-values from the agent
           
            q_values = agent.select_action(state, get_q_values=True)
            
            # Annotate the frame with Q-values
            # Get the next action and state
            action = agent.select_action(state, evaluate_agent=True)
            # import pdb; pdb.set_trace()
            next_state, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            frames = annotate_frame(frames, q_values, obs,current_reward=reward, sum_rewards=episode_reward, frame_number=steps)
            
            cv2.imwrite(f'{path}/frame_image_{steps}.png', cv2.cvtColor(frames[steps-1], cv2.COLOR_RGB2BGR))
            
            state = next_state["state"]
            obs = next_state["pixels"]
            
            if done:
                break

            plt.close()
       
        # Create a video from frames
        # import pdb; pdb.set_trace()
        out = cv2.VideoWriter(os.path.join(path, f'episode_{episode}_reward_{episode_reward:.2f}.avi'), cv2.VideoWriter_fourcc(*'XVID'), 1, (frames[0].shape[1], frames[0].shape[0]))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
