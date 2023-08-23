import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import utils
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
        # self.critic = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=1),  # Adjust the number of channels and kernel size
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 1, kernel_size=1)
        # )

    def forward(self, x):
        x = self.actor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.shit(x)
        # critic_output = self.critic(x)
        return x.reshape(3, 12), None


# A3C Worker Process
class A3CWorker(mp.Process):
    def __init__(self, rank, shared_model, optimizer, env, gamma, step):
        super(A3CWorker, self).__init__()
        self.rank = rank
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.env = env
        self.step_max = step
        self.gamma = gamma
        self.step = 1

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
            while not done:
                # Sample action from policy
                action_prob, _ = model(state)
                action = sample_action(action_prob)  # Implement action sampling
                # Interact with environment


                reward_info = env.take_action(action, self.step)  # Get reward information

                next_state = torch.tensor(env.get_environment(self.step + 1), dtype=torch.float32)
                next_state = next_state.unsqueeze(0).unsqueeze(0)
                reward = torch.tensor(reward_info, dtype=torch.float32).sum()
                Total_Reward += float(reward.float())
                # Compute advantage and TD error
                value, _ = model(state)
                next_value, _ = model(torch.tensor(next_state))
                advantage = reward + gamma * next_value - value
                critic_loss = advantage ** 2

                # Compute policy loss
                def log_prob():
                    _sum = 0
                    apn = action_prob_normalize(action_prob)
                    for i in range(3):
                        for ii in range(env.get_matrix_size()):
                            _sum += apn[i][ii]*action[i][ii]
                    return _sum

                actor_loss = -log_prob() * advantage.detach()

                # Backpropagate and update model
                self.optimizer.zero_grad()
                sum(sum(actor_loss)).backward()
                sum(sum(critic_loss)).backward()
                self.optimizer.step()

                state = next_state
                self.step += 1
                done = not (self.step < self.step_max)

            if done:
                utils.write("data/record.csv", "a", "%f, %f\n" % (Total_Reward, sum(sum(actor_loss)).item()))
                break


# Example environment creator
def self_env():
    import game.factory as factory
    return factory


def action_prob_normalize(action_prob):
    # Normalize action_prob matrix to probabilities
    action_p = action_prob - action_prob.min()
    return action_p / action_p.sum()

def sample_action(action_prob):

    # Normalize action_prob matrix to probabilities
    action_prob_normalized = action_prob_normalize(action_prob)

    # Generating random indices based on the normalized action probabilities
    num_samples = 3 * 12
    random_indices = torch.multinomial(action_prob_normalized.view(-1), num_samples, replacement=True)

    # Reshaping and scaling the random indices to [0, 1000] range
    random_indices_reshaped = random_indices.view(3, 12)
    random_values = random_indices_reshaped * (10 / (action_prob.shape[1] - 1))
    return random_values.int()

# Run the A3C training loop with the new environment
# Main Training Loop
if __name__ == '__main__':

    open("data/record.csv", "w")
    utils.write("data/record.csv", "w", "Total_Reward,Current_Loss\n")
    # Initialize shared model and optimizer
    env = self_env()

    learning_rate = 0.001
    Total_Step = 30
    gamma = 0.7
    num_episodes = 10000000
    checkpoint_interval = 100

    n = env.get_matrix_size()
    shared_model = ActorCritic(8 * n, 3 * n)  # Define the input and output sizes
    shared_model.share_memory()
    optimizer = optim.Adam(shared_model.parameters(), lr=learning_rate)

    mp.set_start_method("spawn")

    # Create worker processes
    num_processes = mp.cpu_count() + torch.cuda.device_count() if torch.cuda.is_available() else 0
    processes = []

    try:
        for episode in range(num_episodes):

            print("Ep: ", episode)

            for rank in range(num_processes):
                p = A3CWorker(rank, shared_model, optimizer, env, gamma, Total_Step)
                p.run()
                processes.append(p)

            if episode % checkpoint_interval == 0:
                checkpoint_path = f'./model/checkpoint_{episode}.pth'
                torch.save({
                    'episode': episode,
                    'model_state_dict': shared_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Saved checkpoint at episode {episode}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        torch.save({
            'episode': episode,
            'model_state_dict': shared_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, './model/interrupted_checkpoint.pth')

    for p in processes:
        p.join()
