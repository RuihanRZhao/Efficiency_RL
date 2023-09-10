import os
from typing import Dict, Union

import torch
from torch import multiprocessing as mp
from network import StrategyRL_Network
from src.game.factory import Factory
from torch import optim as optim


class Strategy_Worker(mp.Process):
    def __init__(self, episode: int, process: int,
                 Central_Net, device, optimizers, environment: Factory, start_day: int = 0,
                 step_end: int = 30):
        super(Strategy_Worker, self).__init__()
        self.episode = episode
        self.process = process

        self.environment = environment
        self.Central_Net = Central_Net
        self.start_delta = start_day
        self.step_now = 0
        self.step_end = step_end
        self.device = device
        self.Optimizer: Dict[str, Union[optim.Optimizer]] = optimizers

    def run(self):
        self.work(self.environment)

    def work(self, environment):
        brain = StrategyRL_Network().to(self.device)
        brain.load_state_dict(self.Central_Net)
        # brain.eval()

        # initialize the environment to the date of start
        environment.reset(self.start_delta + self.step_now)
        state: torch.tensor = torch.tensor([]).to(self.device)
        input_size: list = []
        num_actions: int = 0
        state, input_size, num_actions = environment.info()
        state.to(self.device)
        state = state.unsqueeze(0).unsqueeze(0)

        total_Reward: float = 0
        total_Earn: float = 0
        total_Loss: float = 0

        while self.step_now < self.step_end:
            # get and unpack network output
            out_Brain = brain(state)
            action_Out = out_Brain["AO"].to(self.device)
            action_Prb = out_Brain["AP"].to(self.device)
            action_Gen = out_Brain["AG"].to(self.device)
            info_Procs = out_Brain["IP"].to(self.device)

            # make a step forward, unpack returned rewards
            step_earn: torch.tensor = torch.tensor([]).to(self.device)
            step_reward: torch.tensor = torch.tensor([]).to(self.device)
            self.environment.step(
                action=action_Out.tolist(),
                mode="train"
            )

            # Get next state
            next_state: torch.tensor = torch.tensor([]).to(self.device)
            next_state, _, _ = environment.info()
            next_state.to(self.device)
            next_state = next_state.unsqueeze(0).unsqueeze(0)

            # record total reward and earn for one game
            total_Reward += float(torch.sum(step_reward).item())
            total_Earn += float(torch.sum(step_earn).item())

            # identify loss functions and optimize
            # Information Processing

            # Action Generation
            self.Optimizer["AG"].zero_grad()
            loss_AG = -torch.log(action_Out) * step_reward
            loss_AG.backward()
            self.Optimizer["AG"].step()
            # Action Probability
            self.Optimizer["AP"].zero_grad()
            loss_AP = -torch.log(action_Out) * step_earn
            loss_AP.backward()
            self.Optimizer["AP"].step()
            # Action Output

            # step end
            state = next_state
            self.step_now += 1

        os.system("cls")
        print(

            f"EP: {self.episode:10d}-{self.process:1d}\t| total earn: {total_Earn:15.5f}\t| total reward{total_Reward:10.3f}"
    
        )
