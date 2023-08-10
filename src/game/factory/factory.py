# import
from src.game.factory.util import Database as DB
from src.game.factory.util import Initialize_Factory as IF
from src.game.factory.util import factory_util as f_func


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
    if day == 0: return "Error, day 0 cannot be process."

    material_info = []
    for i in Material_List:
        price_trend = (DB.Get_Material_Price(i.name, day) - (DB.Get_Material_Price(i.name, day - 3) if day >= 3 else DB.Get_Material_Price(i.name, 0))) /  (3 if day >= 3 else day)
        material_info.append([
            i.id,
            i.storage,
            i.max_storage,
            DB.Get_Material_Price(i.name, day),
            round(price_trend, 4)
        ])

    producer_info = []
    for i in f_func.Get_producer_table():
        producer_info.append([
            f_func.Get_Material_by_Name(i[0], Material_List).id,
            f_func.Get_Material_by_Name(i[1], Material_List).id,
            i[2]
        ])
    return material_info, producer_info


def take_action(player_actions, day):
    buy_actions = player_actions[0]
    pro_actions = player_actions[1]
    sel_actions = player_actions[2]
    buy_rewards = []
    pro_rewards = []
    sel_rewards = []
    try:
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
    except ValueError:
        max_len = max(max(len(buy_actions), len(pro_actions)), len(sel_actions))
        return [
            [0 for i in range(max_len)] for i in range(3)
        ]

    return [buy_rewards, pro_rewards, sel_rewards]


