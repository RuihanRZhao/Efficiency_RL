
class A3CWorker(mp.Process):
    def __init__(self, rank, shared_model, optimizer, env, gamma, step, ep):
        super(A3CWorker, self).__init__()
        self.rank = rank
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.env = env
        self.step_max = step
        self.gamma = gamma
        self.step = 1
        self.ep = ep

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
            TA_loss = 0
            TC_loss = 0
            print(f"EP: {self.ep}")
            while not done:
                # Sample action from policy
                action_probs, value = model(state)
                action = action_probs.int()
                # Interact with environment

                reward_info = env.take_action(action, self.step)  # Get reward information
                next_state = torch.tensor(env.get_environment(self.step + 1), dtype=torch.float32)
                next_state = next_state.unsqueeze(0).unsqueeze(0)
                reward = torch.tensor(reward_info).sum()
                Total_Reward += float(reward.float())
                # Compute advantage and TD error
                _, next_value = model(torch.tensor(next_state))
                advantage = reward + gamma * next_value - value
                critic_loss = (advantage ** 2)


                entropy = -(action * action_probs).sum()
                policy_loss = -(action * advantage.detach()).mean() - 1 * entropy

                value_loss = critic_loss.mean()


                total_loss = policy_loss + value_loss
                # Backpropagate and update model
                TA_loss += policy_loss.item()
                TC_loss += value_loss.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                print(reward)
                # print("Step: ", self.step,end="|")
                state = next_state
                self.step += 1
                done = not (self.step < self.step_max)

            if done:
                os.system("cls")
                print(f"EP: {self.ep}-{self.rank}, T_Reward: {Total_Reward}, A_Loss: {float(TA_loss)}, C_Loss: {float(TC_loss)}")
                DB.WriteFile("record.csv", [self.ep, self.rank,Total_Reward, float(TA_loss), float(TC_loss)])
                break