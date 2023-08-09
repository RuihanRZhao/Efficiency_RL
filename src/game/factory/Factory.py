# import
import DB
from Material import Material
from Producer import Producer

import Initialize_Factory as IF

print("Initialize global value...")

# register all material and producer
reward = 0
# Setting Args: about the game
period = 1

total_reward = 0

demo_actions = [
    [40, 40, 40, 40, 40, 40, 40, 0],
    [1, 2, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

rewards = []

print("Initialize global value.... Done")
print("Initialize Database...")

# Initial factory data in SQL
IF.Initialize_Factory_DB()
Material_List = IF.Initialize_Material()
Producer_List = IF.Initialize_Producer()

print("Initialize Database........ Done")

def get_environment(day):

    return
def take_action(player_actions, day):
    buy_actions = player_actions[0]
    pro_actions = player_actions[1]
    sel_actions = player_actions[2]
    buy_rewards = []
    pro_rewards = []
    sel_rewards = []
    # buy
    for i1 in range(len(Material_List)):
        r, _ = Material_List[i1].buy(buy_actions[i1], day)
        buy_rewards.append(r)

    # produce
    for i1 in range(len(Producer_List)):
        pro_rewards.append(Producer_List[i1].produce(pro_actions[i1], Material_List))

    # sell
    for i1 in range(len(Material_List)):
        r, _ = Material_List[i1].sell(sel_actions[i1], day)
        sel_rewards.append(r)

    return [buy_rewards, pro_rewards, sel_rewards]


