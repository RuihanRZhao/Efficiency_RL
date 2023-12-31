import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from factory import Database as DB
import os
from factory import factory
from neuralnet import ActorCritic
from worker import A3CWorker

# Run the A3C training loop with the new environment
# Main Training Loop
if __name__ == '__main__':
    # DB.WriteFile("record.csv", ["Episode", "Actor", "Total_Earn", "Actor_Loss", "critic_Loss"])
    # Initialize shared model and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = factory.factory()

    learning_rate = 0.1
    Total_Step = 2
    gamma = 0.7
    num_episodes = 10000000
    checkpoint_interval = 10000

    pth_code = 20295

    n = env.get_matrix_size()
    shared_model = ActorCritic(8 * n, 3 * n).to(device)  # Define the input and output sizes
    # shared_model.load_state_dict(torch.load(f"model/value/checkpoint_{pth_code}.pth"))
    shared_model.load_state_dict(torch.load(f"model/value/checkpoint_{pth_code}_Interrupt.pth"))
    shared_model.share_memory()
    optimizer = optim.Adam(shared_model.parameters(), lr=learning_rate)

    mp.set_start_method("spawn")

    # Create worker processes
    num_processes = mp.cpu_count()

    processes = []
    try:
        for episode in range(pth_code + 1, num_episodes):
            processes = []
            for rank in range(num_processes):
                p = A3CWorker(rank, shared_model, optimizer, env, gamma, Total_Step, episode, n, device, result_queue)
                p.run()

            if episode % checkpoint_interval == 0:
                checkpoint_path = f'./model/value/checkpoint_{episode}.pth'
                torch.save(shared_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at episode {episode}")

    except KeyboardInterrupt:
        checkpoint_path = f'./model/value/checkpoint_{episode}_Interrupt.pth'
        print("Training interrupted. Saving current state...")
        torch.save(shared_model.state_dict(), checkpoint_path)

    for i in processes:
        i.join()
