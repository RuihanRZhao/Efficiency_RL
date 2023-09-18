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
                 step_end: int = 30, Net_structure: dict = None):
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
        self.brain_structure = Net_structure

    def run(self):
        self.work(self.environment)

    def work(self, environment):
        brain = StrategyRL_Network(
            input_H_size=self.brain_structure["_input_H_size"], input_V_size=self.brain_structure["_input_V_size"],
            num_actions=self.brain_structure["_num_actions"],
            num_action_choice=self.brain_structure["_num_action_choice"],
            IP_hidden_size=self.brain_structure["_IP_hidden_size"], AG_hidden_size=self.brain_structure["_AG_hidden_size"], AP_hidden_size=self.brain_structure["_AP_hidden_size"],
            IP_num_layers=self.brain_structure["_IP_num_layers"], AG_num_layers=self.brain_structure["_AG_num_layers"]
    ).to(self.device)
        brain.load_state_dict(self.Central_Net.state_dict())

        # initialize the environment to the date of start
        environment.reset(self.start_delta + self.step_now)
        state, input_size, num_actions = environment.info()
        state = state.to(self.device)
        # summary variables
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

            step_result = self.environment.step(
                action=action_Out.tolist()[0],
                mode="train"
            )
            step_earn: torch.tensor = step_result["total_earn"].to(self.device)
            step_reward: torch.tensor = step_result["total_reward"].to(self.device)

            # Get next state
            next_state, _, _ = environment.info()
            next_state = next_state.to(self.device)

            # record total reward and earn for one game
            total_Reward += float(torch.sum(step_reward).item())
            total_Earn += float(torch.sum(step_earn).item())

            # identify loss functions and optimize
            # Information Processing

            # loss function
            # loss_AG = (step_reward+torch.log(action_Out)).sum()
            loss_AP = (step_earn+torch.pow(action_Out,step_earn)).sum()

            # zero grad
            self.Optimizer["All"].zero_grad()
            # self.Optimizer["AP"].zero_grad()

            # backward
            # loss_AG.backward(retain_graph=True)
            loss_AP.backward()

            self.Optimizer["All"].step()

            # self.Optimizer["AP"].step()

            # step end
            state = next_state
            self.step_now += 1

        print(

            f"EP: {self.episode:10d}-{self.process:1d}\t| total earn: {total_Earn:15.3f}\t| total reward{total_Reward:20.3f}\t"
            f"Out: {action_Out.tolist()[0]}"
    
        )
