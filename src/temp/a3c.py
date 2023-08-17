import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import Database as DB
import os

# from src.NN_Model.a3c_torch.unet import UNet


# Define Actor-Critic Network
# class ActorCritic(nn.Module):
#     def __init__(self, input_size, output_size):  # 8 * n | 3 * n
#         super(ActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=1),  # Adjust the number of channels and kernel size
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(64, output_size, kernel_size=1)
#         )
#         self.critic = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=1),  # Adjust the number of channels and kernel size
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=1)
#         )
#
#     def forward(self, x):
#         x = x.squeeze(0)
#         x = x.reshape(1, -1)
#         actor_output = self.actor(x)
#         critic_output = self.critic(x)
#         return actor_output, critic_output


class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):  # 8 * n | 3 * n
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # Adjust the number of channels and kernel size
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shit = nn.Sequential(
            nn.Linear(128 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 12),
        )
        self.critic = nn.Sequential(
            nn.Linear(128 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Output a single value for the value function
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.actor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        actor_output = self.shit(x)
        value = self.critic(x)

        return actor_output.reshape(3, 12), value


# A3C Worker Process
class A3CWorker(mp.Process):
    def __init__(self, rank, shared_model, optimizer, env, gamma, step, ep):
        super(A3CWorker, self).__init__()
        self.rank = rank
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.env = env
        self.step_max = step
        self.gamma = gamma
        self.step = 1
        self.ep = ep

    def run(self):
        self.worker(self.env, self.gamma)

    def worker(self, env, gamma):
        model = ActorCritic(0, 0)  # Define the input and output sizes

        # Synchronize model with shared model
        # model.load_state_dict(self.shared_model.state_dict())
        model.eval()

        while True:
            # Perform A3C training steps
            state = torch.tensor(env.get_environment(self.step), dtype=torch.float32)  # Get the 8*n matrix
            state = state.unsqueeze(0).unsqueeze(0)
            done = False
            Total_Reward = 0
            TA_loss = 0
            TC_loss = 0
            print(f"EP: {self.ep}")
            while not done:
                # Sample action from policy
                action_probs, value = model(state)
                action = action_probs.int().to(0)
                # Interact with environment

                reward_info = env.take_action(action, self.step)  # Get reward information
                next_state = torch.tensor(env.get_environment(self.step + 1), dtype=torch.float32)
                next_state = next_state.unsqueeze(0).unsqueeze(0).to(0)
                reward = torch.tensor(reward_info).sum()
                Total_Reward += float(reward.float())
                # Compute advantage and TD error
                _, next_value = model(torch.tensor(next_state))
                advantage = reward + gamma * next_value - value
                critic_loss = (advantage ** 2)


                entropy = -(action * action_probs).sum()
                policy_loss = -(action * advantage.detach()).mean() - 1 * entropy

                value_loss = critic_loss.mean()


                total_loss = policy_loss + value_loss
                # Backpropagate and update model
                TA_loss += policy_loss.item()
                TC_loss += value_loss.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                print(action_probs)
                # print("Step: ", self.step,end="|")
                state = next_state
                self.step += 1
                done = not (self.step < self.step_max)

            if done:
                os.system("clear")
                print(f"EP: {self.ep}-{self.rank}, T_Reward: {Total_Reward}, A_Loss: {float(TA_loss)}, C_Loss: {float(TC_loss)}")
                DB.WriteFile("record.csv", [self.ep, self.rank,Total_Reward, float(TA_loss), float(TC_loss)])
                break


# Example environment creator
def self_env():
    import factory
    return factory


# Run the A3C training loop with the new environment
# Main Training Loop
if __name__ == '__main__':
    DB.WriteFile("record.csv", ["Episode", "Actor", "Total_Reward", "Actor_Loss", "critic_Loss"])
    # Initialize shared model and optimizer
    env = self_env()

    learning_rate = 0.001
    Total_Step = 30
    gamma = 0.7
    num_episodes = 10000000
    checkpoint_interval = 1000000

    n = env.get_matrix_size()
    shared_model = ActorCritic(8 * n, 3 * n)  # Define the input and output sizes
    shared_model.share_memory()
    optimizer = optim.Adam(shared_model.parameters(), lr=learning_rate)

    mp.set_start_method("spawn")

    # Create worker processes
    num_processes = mp.cpu_count()
    processes = []

    try:
        for episode in range(num_episodes):
            for rank in range(num_processes):
                p = A3CWorker(rank, shared_model, optimizer, env, gamma, Total_Step, episode)
                p.run()
                processes.append(p)

            if episode % checkpoint_interval == 0:
                checkpoint_path = f'./model/checkpoint_{episode}.pth'
                torch.save(shared_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at episode {episode}")
    except KeyboardInterrupt:
        checkpoint_path = f'./model/checkpoint_{episode}_Interrupt.pth'
        print("Training interrupted. Saving current state...")
        torch.save(shared_model.state_dict(), checkpoint_path)

    for p in processes:
        p.join()