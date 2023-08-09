# System Part
import os

# Pytorch Part
import torch
import torch.nn.functional as function


os.environ["OMP_NUM_THREADS"]="1"

class Actor(object):
    def __init__(self, model, env, args, state, gpu):
        self.model = model      # A3C Model, include Actor and Critic Network
        self.env = env          # For connecting with outside, to get state and do actions
        self.state = state      #
        self.args = args        # arguments for training
        self.gpu = gpu       # limit which gpu to use

        # Total states in EPs
        self.log_Prob = []
        self.values = []
        self.rewards = []
        self.entropies = []

        # Current EP
        self.eps_length = 0
        self.if_done = True
        self.info = None        # extra data for current EP
        self.reward = 0

        # LSTM args
        self.hx = None
        self.cx = None
        self.hidden_size = args.hiddden_size

    def train(self):
        # get forward states
        value, logit, self.hx, self.cx = self.model(self.state.unsqueeze(0), self.hx, self.cx)

        # calculate action prob and log prob
        prob = function.softmax(logit, dim=1)
        log_prob = function.log_softmax(logit, dim=1)

        # entropy of action
        entropy = -(log_prob * prob).sum(1)

        # choose action by prob
        action = prob.multinomial(1).data

        # choose action by log_prob
        log_action = log_prob.gather(1,action)

        # take action and get states for next step or get the stop info
        state, self.reward, self.if_done, self.info = self.env.step(action.item())

        if self.gpu >= 0:
            with torch.cuda.device(self.gpu): self.state = torch.from_numpy(state).float().cuda()
        else:
            self.state = torch.from_numpy(state).float()

        # ready for next EP
        self.eps_length += 1
        self.reward = max(min(self.reward, 1), -1)
        # store new data
        self.values.append(value)
        self.log_Prob.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def test(self):
        with torch.no_grad():
            if self.if_done:
                if self.gpu >= 0:
                    with torch.cuda.device(self.gpu):
                        self.cx = torch.zeros(1, self.hidden_size).cuda()
                        self.hx = torch.zeros(1, self.hidden_size).cuda()
                else:
                    self.cx = torch.zeros(1, self.hidden_size)
                    self.hx = torch.zeros(1, self.hidden_size)

            value, logit, self.hx, self.cx = self.model(
                self.state.unsqueeze(0), self.hx, self.cx
            )
            prob = function.softmax(logit, dim=1)
            action = prob.cpu().numpy().argmax()
        state, self.reward, self.if_done, self.info = self.env.step(action)
        if self.gpu >= 0:
            with torch.cuda.device(self.gpu):
                self.state = torch.from_numpy(state).float().cuda()
        else:
            self.state = torch.from_numpy(state).float()

        self.eps_length += 1
        return self

    def clear(self):
        self.values = []
        self.log_Prob = []
        self.rewards = []
        self.entropies = []
        return self

