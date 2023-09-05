# python standard
# JSON support to load the database info
import json
from typing import Dict, Union

# components of the factory
from .object import Material, Producer, Obj_Initial
from .tool_data import SQL

# pytorch
import torch

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


class Factory:
    def __init__(self, date_plus, date_period):
        """
        Initialize a factory environment.

        Args:
            date_plus (int): The number of days to advance the factory's date.
            date_period (int): The number of days in the factory's date period.

        Attributes:
            materials (list[Material]): A list of Material objects representing materials in the factory.
            producers (list[Producer]): A list of Producer objects representing producers in the factory.
            raw (dict): A dictionary containing raw data for materials and producers.
            database (SQL): An SQL object for connecting to a MySQL database.
            obj_ini (Obj_Initial): An Obj_Initial object for initializing raw data.
        """
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
        """
        Reset the factory by restoring materials and producers to their initial state.
        """
        self.materials = self.raw["material"]
        self.producers = self.raw["producer"]

    def info(self) -> tuple[torch.tensor, list[int]]:
        """
        Get information about the factory's materials and producers.

        Returns:
            tuple[torch.tensor, list[int]]: A tuple containing the environment tensor and matrix size.
        """
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

    def step(self, action: list[float], mode: str = "train") -> Dict[str, torch.tensor]:  # make one step forward
        """
        Take one step forward in the factory environment.

        Args:
            action (list[float]): A list of actions to be performed.
            mode (str, optional): The mode in which the factory is running ("train" or "play"). Default is "train".

        Returns:
            Dict[str, torch.tensor]: A dictionary containing relevant information based on the chosen mode.
        """
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

        # choose return values by mode choice

        # return when train mode
        def train_return() -> Dict[str, torch.tensor]:
            total_earn = torch.tensor(trade_earn + produce_earn)
            total_reward = torch.tensor(trade_reward + produce_reward)
            return {
                "total_earn": total_earn,
                "total_reward": total_reward
            }

        # return when play mode
        def play_return() -> Dict[str, torch.tensor]:
            total_earn = torch.tensor(trade_earn + produce_earn)
            total_reward = torch.tensor(trade_reward + produce_reward)
            total_output = torch.tensor(trade_output + produce_output)
            return {
                "total_earn": total_earn,
                "total_reward": total_reward,
                "total_output": total_output
            }

        # match dictionary
        switch = {
            "train": train_return(),
            "play": play_return()
        }

        return switch[mode]

if __name__ == '__main__':
    # a demo to test info and step
    pass

