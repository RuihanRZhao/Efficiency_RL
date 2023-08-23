# import
import torch

from game.factory.util import Database as DB
from game.factory.util import Initialize_Factory as IF
from game.factory.util import factory_util as f_func

import numpy as np

# set numpy to display without using scientific notation
np.set_printoptions(suppress=True)

# print("Initialize global value...")

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

# print("Initialize global value.... Done")
# print("Initialize Database...")
# Initial factory data in SQL
IF.Initialize_Factory_DB()
# print("Initialize Database........ Done")
# print("Get Material Data  ........ ")
Material_List = IF.Initialize_Material()
# print("Get Material Data  ........ Done")
# print("Get Producer Data  ........ ")
Producer_List = IF.Initialize_Producer()


# print("Get Producer Data  ........ Done")


def get_environment(day):
    if day == 0: return "Error, day 0 cannot be process."

    p_info = f_func.Get_producer_table()

    M_len = len(Material_List)
    P_len = len(p_info)

    material_info = np.zeros((5, M_len))

    for i in range(M_len):
        price_trend = (
                              DB.Get_Material_Price(Material_List[i].name, day) -
                              (DB.Get_Material_Price(Material_List[i].name,
                                                     day - 3) if day >= 3 else DB.Get_Material_Price(
                                  Material_List[i].name, 0))) / \
                      (3 if day >= 3 else day
                       )

        temp = np.array([
            Material_List[i].id,
            Material_List[i].storage,
            Material_List[i].max_storage,
            DB.Get_Material_Price(Material_List[i].name, day),
            round(price_trend, 4)
        ])

        for i1 in range(5):
            material_info[i1, i] = temp[i1]

    producer_info = np.zeros((3, P_len))
    for i in range(P_len):
        temp = np.array([
            f_func.Get_Material_by_Name(p_info[i][0], Material_List).id,
            f_func.Get_Material_by_Name(p_info[i][1], Material_List).id,
            p_info[i][2]
        ])
        for i1 in range(3):
            producer_info[i1, i] = temp[i1]

    info = np.zeros((8, M_len if M_len > P_len else P_len), dtype=np.float32)
    info[:5, :M_len] = material_info
    info[5:8, :P_len] = producer_info
    return info


def take_action(player_actions, day):
    buy_actions = player_actions[0]
    pro_actions = player_actions[1]
    sel_actions = player_actions[2]
    buy_rewards = []
    pro_rewards = []
    sel_rewards = []

    print(player_actions)

    # buy
    for i1 in range(len(Material_List)):
        buy_rewards.append(Material_List[i1].buy(int(buy_actions[i1]), day))

    # produce
    count = 0
    for i1 in range(len(Producer_List)):
        pro_rewards.append(Producer_List[i1].produce(int(pro_actions[i1]), Material_List))

    # sell

    count = 0
    for i1 in range(len(Material_List)):
        sel_rewards.append(Material_List[i1].sell(int(sel_actions[i1]), day))


    info = np.zeros((3, get_matrix_size()), dtype=np.float32)

    info[0, :len(buy_rewards)] = buy_rewards
    info[1, :len(pro_rewards)] = pro_rewards
    info[2, :len(sel_rewards)] = sel_rewards


    # print("REWARD: ", info)

    return info


def get_matrix_size():
    return DB.check_max_table_lenght()
