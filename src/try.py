import torch
import game.factory as fac
from NN_Model.a3c_torch.v_0_0_1.AC import ActorCritic  # Import your ActorCritic class
import NN_Model.a3c_torch.v_0_0_1.func as f

def test_model(model, env, num_episodes=10):
    total_rewards = []
    model.eval()
    for episode in range(num_episodes):

        step = 1
        state = torch.tensor(env.get_environment(step), dtype=torch.float32)  # Get the 8*n matrix
        state = state.unsqueeze(0).unsqueeze(0)
        done = False
        episode_reward = 0
        print("Start")
        while step < 31:
            print(f"Episode {episode}, Step {step}")
            action_prob, _ = model(state)

            action = f.sample_action(action_prob)

            reward_info = env.take_action(action, step)  # Get reward information
            next_state = torch.tensor(env.get_environment(step + 1), dtype=torch.float32)
            next_state = next_state.unsqueeze(0).unsqueeze(0)
            reward = torch.tensor(reward_info, dtype=torch.float32).sum()
            episode_reward += reward
            state = next_state
            step += 1

        total_rewards.append(episode_reward)
        print(f"Reward: {episode_reward}")

    average_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")



if __name__ == '__main__':
    # Load the trained model
    model = ActorCritic(8*12, 3*12)  # Define input_size and output_size
    model.load_state_dict(torch.load("./NN_Model/a3c_torch/model/Interrput_checkpoint_524939.pth")["model_state_dict"])  # Load your trained model's state dict

    # Create the environment
    env = fac  # Initialize your environment

    # Test the model
    test_model(model, env)