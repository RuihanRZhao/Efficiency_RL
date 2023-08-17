import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import utils
import temp.factory as fac
from NN_Model.a3c_torch.v_0_0_1.worker import A3CWorker
from NN_Model.a3c_torch.v_0_0_1.AC import ActorCritic
import NN_Model.a3c_torch.v_0_0_1.func as f
import temp.Database as DB

if __name__ == '__main__':
    DB.WriteFile("record.csv", "Total_Reward,Current_Loss\n")
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
                torch.save(shared_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint at episode {episode}")
    except KeyboardInterrupt:
        checkpoint_path = f'./model/checkpoint_{episode}_Interrupt.pth'
        print("Training interrupted. Saving current state...")
        torch.save(shared_model.state_dict(), checkpoint_path)

    for p in processes:
        p.join()