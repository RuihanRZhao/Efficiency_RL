import torch
import torch.multiprocessing as mp
from neuralnet import ActorCritic
from factory import Fac_Value as FV
import numpy as np
import os
from factory import Database as DB
import torch.nn as nn


class A3CWorker(mp.Process):
    def __init__(self, rank, shared_model, optimizer, env, gamma, step, ep, n, device, result_queue):
        super(A3CWorker, self).__init__()
        self.rank = rank
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.env = env
        self.step_max = step
        self.gamma = gamma
        self.step = 1
        self.ep = ep
        self.n = n
        self.device = device
        self.rq = result_queue

    def run(self):
        self.worker(self.env, self.gamma)

    def worker(self, env, gamma):
        model = ActorCritic(8 * self.n, 3 * self.n).to(self.device)  # Define the input and output sizes

        # Synchronize model with shared model
        # model.load_state_dict(self.shared_model.state_dict())

        while True:
            # Perform A3C training steps
            state = torch.tensor(env.get_environment(self.step), dtype=torch.float32).to(
                self.device)  # Get the 8*n matrix
            state = state.unsqueeze(0).unsqueeze(0)
            done = False
            Total_Earn = 0
            Total_Act = 0
            To_loss = 0
            while not done:
                # Sample action from policy
                action_probs, value_G = model(state)
                action = action_probs.int()

                # Interact with environment

                reward_info, value_info = env.take_action(action, self.step)  # Get reward information
                next_state = torch.tensor(env.get_environment(self.step + 1), dtype=torch.float32).to(self.device)
                next_state = next_state.unsqueeze(0).unsqueeze(0)
                reward = torch.tensor(reward_info, requires_grad=True).to(self.device)
                value = torch.tensor(value_info, requires_grad=True).to(self.device)
                Earn = float(value.sum().float())
                Act = float(reward.sum().float())
                # print(
                #     f"Earn: {Earn: 20.4f}\t\t"
                #     f"Reward: {float(reward.sum().float()): 20.4f}"
                # )
                Total_Earn += Earn
                Total_Act += Act
                def no_inf(tensor_with_inf):
                    max_float = torch.finfo(tensor_with_inf.dtype).max/7
                    return torch.where(tensor_with_inf == float('inf'), max_float, tensor_with_inf)

                # Compute advantage and TD error
                T_ = (-reward+1)*value
                T_loss = no_inf(100 * torch.pow(0.99, T_)).sum()

                # Action loss
                A_loss = no_inf(100 * torch.pow(0.99, reward)).sum()

                To_loss += A_loss
                # Backpropagate and update model
                self.optimizer.zero_grad()
                A_loss.backward()
                self.optimizer.step()

                # T_loss use
                # print(f"step: {self.ep:10d}-{self.rank:1d}-{self.step}\tEarn: {Earn:15.5f}\tAct: {Act:15.5f}\t\t  Loss:{float(T_loss):100.20f}")

                # A_loss use
                print(f"step: {self.ep:10d}-{self.rank:1d}-{self.step}\tEarn: {Earn:15.5f}\tAct: {Act:15.5f}\t\t  Loss:{float(A_loss):40.20f}")

                state = next_state
                self.step += 1
                done = not (self.step < self.step_max)

            if done:
                os.system("cls")
                print(

                    f"EP: {self.ep:10d}-{self.rank:1d}\tT_Earn: {Total_Earn:15.5f}\tAct_Score{Total_Act:10.3f}\t\t\tTotal Loss: {float(To_loss):100.20f}"

                    f"\n------------------------------------------------------------------------------------------------------------------------\n"
                    # f"EP: {self.ep:10d}-{self.rank:1d}\tT_Earn: {Total_Earn:15.5f}\tA_Loss: {float(TA_loss):10.2f}\tValue_Loss: {float(TC_loss):20.5f}"
                )

                # DB.WriteFile("record.csv", [self.ep, self.rank, Total_Earn, float(TA_loss), float(TC_loss)])
                break

