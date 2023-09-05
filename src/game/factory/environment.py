# python standard
# JSON support to load the database info
import json
from typing import Dict, Union

# pytorch
import torch
# components of the factory
from .object import Material, Producer, Obj_Initial
from .tool_data import SQL


# load database information
def _load_database_info():
    with open('.database', 'r') as json_file:
        # Load the JSON data into a Python dictionary
        # {
        #     "host": "localhost",
        #     "port": 666,
        #     "user": "username",
        #     "password": "password"，
        #     "database": "DB"
        # }

        return json.load(json_file)


class factory:
    def __init__(self, date_plus, date_period):
        # factory inner data in gaming
        self.materials: list[Material] = []
        self.producers: list[Producer] = []

        # origin data
        self.raw = {
            "material": list[Material],
            "producer": list[Producer],
        }
        # other data that will be used in the environment
        # connect mySQL database
        _database_info = _load_database_info()
        self.database = SQL(
            host=_database_info["host"],
            port=_database_info["port"],
            user=_database_info["user"],
            password=_database_info["password"],
            database=_database_info["database"]
        )
        # pass database to obj_initial get the raw data of material and producer
        self.obj_ini = Obj_Initial(self.database)
        # get the raw data
        self.raw["material"] = self.obj_ini.material_initialize()
        self.raw["producer"] = self.obj_ini.producer_initialize()
        # initialize
        self.reset()

    def reset(self) -> None:
        self.materials = self.raw["material"]
        self.producers = self.raw["producer"]

    def info(self) -> tuple(torch.tensor, list[int]):
        mat_info = []
        pro_info = []
        # generate the material data matrix
        for item in self.materials:
            mat_info.append([
                int(item.un_id),
                item.inventory,
                item.inventory_cap,
                item.cache,
                item.cache_cap,
                1 if item.trade_permit["purchase"] else 0,
                1 if item.trade_permit["sale"] else 0,
                item.price["price_now"],
            ])

        # generate the producer data matrix
        for item in self.producers:
            for mat_key, mat_value in item.material.items():
                pro_info.append([
                    int(item.un_id),
                    item.daily_low_cost,
                    item.daily_produce_cap,
                    int(mat_key),
                    mat_value,
                ])

        print(mat_info, "\n", pro_info)

        mat_count = len(mat_info)
        pro_count = len(pro_info)
        mat_colum = len(mat_info[0])
        pro_colum = len(pro_info[0])
        matrix_size = [mat_colum + pro_colum, mat_count if mat_count > pro_count else pro_count]

        # transfer list matrix into tensor

        def write_tensor(target: torch.tensor, matrix: list[list], m_count: int, m_colum: int, start: list):
            for m_c in range(m_count):
                for m_l in range(m_colum):
                    target[m_l + start[0], m_c + start[1]] = matrix[m_c][m_l]

        env_tensor = torch.zeros(matrix_size)
        write_tensor(env_tensor, mat_info, mat_count, mat_colum, [0, 0])
        write_tensor(env_tensor, pro_info, pro_count, pro_colum, [mat_colum + 1, 0])

        return env_tensor, matrix_size

    def step(self, action: list[float]) -> torch.tensor:  # make one step forward
        # action amount needs
        mat_act_count = len(self.materials)
        pro_act_count = len(self.producers)

        # record actions
        trade_action = action[:mat_act_count]
        produce_action = action[mat_act_count:]

        # record result
        trade_result: list[Dict[str, Union[int, float, str]]] = []
        produce_result: list[Dict[str, Union[int, float, str]]] = []

        # trade
        for act in range(mat_act_count):
            trade_result.append(
                self.materials[act].trade(trade_action[act])
            )

        # produce
        for act in range(pro_act_count):
            produce_result.append(
                self.producers[act].produce(produce_action[act])
            )

        # get result unpacked
        def unpack_result(target: list[Dict[str, Union[int, float, str]]]):
            Earn: list[float] = []
            Reward: list[float] = []
            Output: list[str] = []

            for item in target:
                Earn.append(item["Earn"])
                Reward.append(item["Reward"])
                Output.append(item["Output"])

            return Earn, Reward, Output

        trade_earn, trade_reward, trade_output = unpack_result(trade_result)
        produce_earn, produce_reward, produce_output = unpack_result(produce_result)







if __name__ == '__main__':
    pass
