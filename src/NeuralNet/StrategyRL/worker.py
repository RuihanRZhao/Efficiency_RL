import torch
from torch import multiprocessing as mp
from network import StrategyRL_Network
from src.game.factory import Factory



class Strategy_Worker(mp.Process):
    def __init__(self, Central_Net, device, environment: Factory = Factory(), start_day: int = 0, step_end: int = 30):
        super(Strategy_Worker, self).__init__()
        self.environment = environment
        self.Central_Net = Central_Net
        self.start_delta = start_day
        self.step_now = 0
        self.step_end = step_end
        self.device = device

    def run(self):
        self.work(self.environment)

    def work(self, environment):
        brain = StrategyRL_Network().to(self.device)
        brain.load_state_dict(self.Central_Net)
        brain.eval()

        # initialize the environment to the date of start
        environment.reset(self.start_delta + self.step_now)
        while True:
            state: torch.tensor = torch.tensor([]).to(self.device)
            input_size: list = []
            num_actions: int = 0
            state, input_size, num_actions = environment.info()
            state = state.unsqueeze(0).unsqueeze(0)

            total_Reward: float = 0
            total_Earn: float = 0
            total_Loss: float = 0

            while self.step_now < self.step_end:
                out_Brain = brain(state)
                action_Out = out_Brain["AO"].to(self.device)
                action_Prb = out_Brain["AP"].to(self.device)
                action_Gen = out_Brain["AG"].to(self.device)
                info_Procs = out_Brain["IP"].to(self.device)



                self.step_now += 1


        
