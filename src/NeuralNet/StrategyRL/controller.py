import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

# system
import os
from datetime import datetime, timedelta
import random

# environment
from src.game.factory.environment import Factory
from network import StrategyRL_Network as Network_Structure
from worker import Strategy_Worker

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    environment = Factory()

    # training variable
    learning_rate = 0.01
    step_max = 7
    num_episode = 1_000_000_000
    checkpoint_interval = 100_000

    _, matrix_size, num_actions = environment.info()
    input_H_size = matrix_size[1]
    input_V_size = matrix_size[2]

    # Network settings
    Net_setting = {
        "_input_H_size": input_H_size,
        "_input_V_size": input_V_size,
        "_num_actions": num_actions,
        "_num_action_choice": 4,
        "_IP_hidden_size": 16,
        "_AG_hidden_size": 16,
        "_AP_hidden_size": 16,
        "_IP_num_layers": 2,
        "_AG_num_layers": 2,
    }

    central_network = Network_Structure(
        input_H_size=Net_setting["_input_H_size"], input_V_size=Net_setting["_input_V_size"],
        num_actions=Net_setting["_num_actions"],
        num_action_choice=Net_setting["_num_action_choice"],
        IP_hidden_size=Net_setting["_IP_hidden_size"], AG_hidden_size=Net_setting["_AG_hidden_size"], AP_hidden_size=Net_setting["_AP_hidden_size"],
        IP_num_layers=Net_setting["_IP_num_layers"], AG_num_layers=Net_setting["_AG_num_layers"]
    ).to(device)

    model_num = 0
    if model_num != 0:
        central_network.load_state_dict(torch.load(f"model/checkpoint_{model_num}_Interrupt.pth")["State_Dict"])

    central_network.share_memory()
    optimizer = {
        "All": optim.Adam(central_network.parameters(), lr=learning_rate),
        "AG": optim.Adam(central_network.action_generation.parameters(), lr=learning_rate),
        "AP": optim.Adam(central_network.action_probability.parameters(), lr=learning_rate),
        "IP": optim.Adam(central_network.information_processing.parameters(), lr=learning_rate),
    }

    mp.set_start_method("spawn")
    num_processes = mp.cpu_count()

    processes = []

    def net_storage(brain, episode) -> dict:
        package = {
            "Episode": episode,
            "State_Dict": brain.state_dict()
        }
        return package

    try:
        for episode in range(model_num, num_episode):
            processes = []
            for rank in range(num_processes):
                process = Strategy_Worker(
                    episode=episode, process=rank,
                    Central_Net=central_network, device=device,
                    optimizers=optimizer,
                    environment=environment,
                    # start_day=random.randint(30, 200),
                    start_day=30,
                    step_end=step_max,
                    Net_structure=Net_setting,
                )
                process.run()

            if episode % checkpoint_interval == 0:
                checkpoint_path = f'./model/checkpoint_{episode}.pth'
                torch.save(net_storage(central_network, episode), checkpoint_path)
                print(f"Saved checkpoint at episode {episode}")

    except KeyboardInterrupt:
        checkpoint_path = f'./model/checkpoint_{episode}_Interrupt.pth'
        print("Training interrupted. Saving current state...")
        torch.save(net_storage(central_network, episode), checkpoint_path)

    for i in processes:
        i.join()

