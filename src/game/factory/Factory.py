# import
import DB
from Material import Material
from Producer import Producer

import Initialize_Factory as IF

print("Initialize global value...")


# Table structure
Material = ("name", "storage", "Max_Store", "Max_Extra_Store")
Material_Price = ("name", "day", "price")
Producer = ("Produce", "Origin", "Origin_Volume")

# register all material and producer
reward = 0
# Setting Args: about the game
period = 1

total_reward = 0

actions = [
    [40, 40, 40, 40, 40, 40, 40, 0],
    [ 1,  2,  1,  1,  0,  0,  0, 0],
    [ 1,  1,  1,  1,  1,  1,  1, 1]
]

rewards = []

print("Initialize global value.... Done")
print("Initialize Database...")

# Initial factory data in SQL
IF.Initialize_Factory_DB()
Material_List = IF.Initialize_Material()
Producer_List = IF.Initialize_Producer()

print("Initialize Database........ Done")


for day in range(1, period+1):
    print("Total_reward: ", total_reward)

    print("-----------------------------------------")
    buy_actions = actions[0]
    pro_actions = actions[1]
    sel_actions = actions[2]
    buy_rewards = []
    pro_rewards = []
    sel_rewards = []
    # buy
    print('buy')
    for i1 in range(len(Material_List)):
        r, _ = Material_List[i1].buy(buy_actions[i1], day)
        buy_rewards.append(r)

    # produce
    print('produce')
    for i1 in range(len(Producer_List)):
        pro_rewards.append(Producer_List[i1].produce(pro_actions[i1], Material_List))

    # sell
    print('sell')
    for i1 in range(len(Material_List)):
        r, _ = Material_List[i1].sell(sel_actions[i1], day)
        sel_rewards.append(r)


    rewards = [buy_rewards, pro_rewards, sel_rewards]
    # count rewards temp!!!! should be replaced by torch.tensor
    temp = 0
    for i in buy_rewards:
        temp += i
    total_reward += temp
    print("buy_reward:", temp)
    temp = 0
    for i in pro_rewards:
        temp += i
    total_reward += temp
    print("pro_reward:", temp)
    temp = 0
    for i in sel_rewards:
        temp += i
    total_reward += temp
    print("sel_reward:", temp)


print("Mat\tPrice")
for i in Material_List:
    print(i.name, "\t",DB.Get_Material_Price(i.name, 1))

print("= ACTIONS", actions, "\n= REWARDS", rewards, "\nT_r", total_reward)