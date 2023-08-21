import torch
import torch.multiprocessing as mp
from neuralnet import ActorCritic
from factory import Fac_Value as FV
import numpy as np
import os
from factory import Database as DB
import torch.nn as nn


class A3CWorker(mp.Process):
    def __init__(self, rank, shared_model, optimizer, env, gamma, step, ep, n, device):
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

    def run(self):
        self.worker(self.env, self.gamma)

    def worker(self, env, gamma):
        model = ActorCritic(8 * self.n, 3 * self.n).to(self.device)  # Define the input and output sizes

        # Synchronize model with shared model
        # model.load_state_dict(self.shared_model.state_dict())
        model.eval()

        while True:
            # Perform A3C training steps
            state = torch.tensor(env.get_environment(self.step), dtype=torch.float32).to(
                self.device)  # Get the 8*n matrix
            state = state.unsqueeze(0).unsqueeze(0)
            done = False
            Total_Earn = 0
            TA_loss = 0
            TC_loss = 0
            while not done:
                # Sample action from policy
                action_probs, value_G = model(state)
                action = action_probs.int()

                # Interact with environment

                reward_info, value_info = env.take_action(action, self.step)  # Get reward information
                next_state = torch.tensor(env.get_environment(self.step + 1), dtype=torch.float32).to(self.device)
                next_state = next_state.unsqueeze(0).unsqueeze(0)
                reward = torch.tensor(reward_info, requires_grad=True).to(self.device)
                value = torch.tensor(value_info).to(self.device)
                Earn = float(value.sum().float())
                # print(
                #     f"Earn: {Earn: 20.4f}\t\t"
                #     f"Reward: {float(reward.sum().float()): 20.4f}"
                # )

                Total_Earn += Earn
                # Compute advantage and TD error
                mseLoss = nn.MSELoss()
                value_loss = mseLoss(value_G, value.sum())

                target_reward = np.zeros((3, 12), dtype=np.float32)
                target_reward[0, :8] = [2 * FV.Cost_Do_Nothing for i in range(8)]
                target_reward[1, :12] = [3 * FV.Cost_Do_Nothing for i in range(12)]
                target_reward[2, :8] = [3 * FV.Cost_Do_Nothing for i in range(8)]
                target_reward = torch.tensor(target_reward, requires_grad=True).to(self.device)
                policy_loss = ((reward - target_reward)**2).sum()

                TA_loss += policy_loss.item()
                TC_loss += value_loss.item()

                # Backpropagate and update model
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
                # print("Step: ", self.step,end="|")
                state = next_state
                self.step += 1
                done = not (self.step < self.step_max)

            if done:
                # os.system("cls")
                print(
                    f"EP: {self.ep:10d}-{self.rank:1d}\tT_Earn: {Total_Earn:15.5f}\tA_Loss: {float(TA_loss):20.5f}\tC_Loss: {float(TC_loss):20.5f}")
                DB.WriteFile("record.csv", [self.ep, self.rank, Total_Earn, float(TA_loss), float(TC_loss)])
                break
