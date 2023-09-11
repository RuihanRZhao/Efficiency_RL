
class A3CWorker(mp.Process):
    def __init__(self, rank, shared_model, optimizer, env, gamma, step):
        super(A3CWorker, self).__init__()
        self.rank = rank
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.env = env
        self.step_max = step
        self.gamma = gamma
        self.step = 1

    def run(self):
        self.worker(self.env, self.gamma)

    def worker(self, env, gamma):
        model = ActorCritic(0, 0)  # Define the input and output sizes

        # Synchronize model with shared model
        # model.load_state_dict(self.shared_model.state_dict())
        model.eval()

        while True:
            # Perform A3C training steps
            state = torch.tensor(env.get_environment(self.step), dtype=torch.float32)  # Get the 8*n matrix
            state = state.unsqueeze(0).unsqueeze(0)
            done = False
            Total_Reward = 0
            while not done:
                # Sample action from policy
                action_prob, _ = model(state)
                action = sample_action(action_prob)  # Implement action sampling
                # Interact with environment


                reward_info = env.take_action(action, self.step)  # Get reward information

                next_state = torch.tensor(env.get_environment(self.step + 1), dtype=torch.float32)
                next_state = next_state.unsqueeze(0).unsqueeze(0)
                reward = torch.tensor(reward_info, dtype=torch.float32).sum()
                Total_Reward += float(reward.float())
                # Compute advantage and TD error
                value, _ = model(state)
                next_value, _ = model(torch.tensor(next_state))
                advantage = reward + gamma * next_value - value
                critic_loss = advantage ** 2

                # Compute policy loss
                def log_prob():
                    _sum = 0
                    apn = action_prob_normalize(action_prob)
                    for i in range(3):
                        for ii in range(env.get_matrix_size()):
                            _sum += apn[i][ii]*action[i][ii]
                    return _sum

                actor_loss = -log_prob() * advantage.detach()

                # Backpropagate and update model
                self.optimizer.zero_grad()
                sum(sum(actor_loss)).backward()
                sum(sum(critic_loss)).backward()
                self.optimizer.step()

                state = next_state
                self.step += 1
                done = not (self.step < self.step_max)

            if done:
                print("Total Reward: ", Total_Reward, "\tLoss: ", sum(sum(actor_loss)).item())
                # utils.write("Nanjing/record.csv", "a", "%f, %f\n" % (Total_Reward, sum(sum(actor_loss)).item()))
                break
