import os
from typing import Dict, Union

import torch
from torch import multiprocessing as mp
from torch import optim as optim

from network import StrategyRL_Network
from src.game.factory import Factory


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
            IP_hidden_size=self.brain_structure["_IP_hidden_size"],
            AG_hidden_size=self.brain_structure["_AG_hidden_size"],
            AP_hidden_size=self.brain_structure["_AP_hidden_size"],
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

        total_mock_AG_reward = []
        total_mock_AG_earn = []
        record_action_Out = []
        record_action_Prb = []
        record_action_Gen = []
        record_info_Procs = []
        record_state = []
        record_step_earn = []
        record_step_reward = []
        record_prb_act = []
        while self.step_now < self.step_end:
            # get and unpack network output
            out_Brain = brain(state)

            record_state.append(state)

            action_Out = out_Brain["AO"].to(self.device)
            action_Prb = out_Brain["AP"].to(self.device)
            action_Gen = out_Brain["AG"].to(self.device)
            info_Procs = out_Brain["IP"].to(self.device)

            record_action_Out.append(action_Out)
            record_action_Prb.append(action_Prb)
            record_action_Gen.append(action_Gen)
            record_info_Procs.append(info_Procs)

            # make a step forward, unpack returned rewards

            step_result = self.environment.step(
                action=action_Out.tolist()[0],
                mode="train"
            )
            step_earn: torch.tensor = step_result["total_earn"].to(self.device)
            step_reward: torch.tensor = step_result["total_reward"].to(self.device)
            step_earn.to(torch.float32).requires_grad_()

            record_step_earn.append(step_earn)
            record_step_reward.append(step_reward)

            # Get next state
            next_state, _, _ = environment.info()
            next_state = next_state.to(self.device)

            # record total reward and earn for one game
            total_Reward += float(torch.sum(step_reward).item())
            total_Earn += float(torch.sum(step_earn).item())

            # loss function varibles
            mock_AG_earn, mock_AG_reward = self.environment.action_mock(action_Gen[0])
            mock_AG_earn.requires_grad_()
            mock_AG_reward.requires_grad_()
            total_mock_AG_earn.append(mock_AG_earn)
            total_mock_AG_reward.append(mock_AG_reward)

            prb_act = []
            for num_out in range(len(action_Out[0])):
                for num_prb in range(len(action_Prb[0, num_out])):
                    if action_Out[0, num_out] == action_Gen[0, num_out, num_prb]:
                        prb_act.append(num_prb)
            prb_act = torch.tensor(prb_act)
            record_prb_act.append(prb_act)

            # step end
            state = next_state
            self.step_now += 1

        print(
            f"step_loss: {[float(i.sum()) for i in record_step_reward]}"
            f"EP: {self.episode:10d}-{self.process:1d}\t| total earn: {total_Earn:15.3f}\t| total reward{total_Reward:20.3f}\t"
            f"Out: {action_Out.tolist()[0]}"
        )

        def loss_AG():
            loss = []
            for i in range(len(total_mock_AG_earn)):
                step_loss = -total_mock_AG_reward[i].sum()
                loss.append(step_loss)
            return loss

        _loss_AG = loss_AG()

        def loss_AP():
            loss = []
            for i in range(len(record_step_earn)):
                step_loss = -torch.log(record_prb_act[i])*record_step_earn[i]
                # print(record_prb_act[i], " | ", record_step_earn[i])
                loss.append(step_loss)

            return loss

        _loss_AP = loss_AP()


        self.Optimizer["AG"].zero_grad()
        ag = torch.stack(_loss_AG).sum()
        ag.backward(retain_graph=True)
        self.Optimizer["AG"].step()

        self.Optimizer["AP"].zero_grad()
        torch.stack(_loss_AP).sum().backward()
        self.Optimizer["AP"].step()

        # self.Optimizer["IP"].zero_grad()
        # loss_IP.backward()
        # self.Optimizer["IP"].step()
