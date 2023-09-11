# python standard
# JSON support to load the database info
from typing import Dict, Union
from datetime import datetime, timedelta

# components of the factory
from .object import Material, Producer, Obj_Initial
from src.game.factory.object.tool_data import SQL

# pytorch
import torch


# load database information
def _load_database_info():
    return{
        "host": "localhost",
        "port": 114,
        "user": "reader",
        "password": "666666",
        "database": "Factory"
    }


class Factory:
    def __init__(self):
        """
        Initialize a factory environment.

        :param int day_plus: The number of days to advance the factory's date.

        :ivar list[Material] materials: A list of Material objects representing materials in the factory.
        :ivar list[Producer] producers: A list of Producer objects representing producers in the factory.
        :ivar dict raw: A dictionary containing raw Nanjing for materials and producers.
        :ivar SQL database: An SQL object for connecting to a MySQL database.
        :ivar Obj_Initial obj_ini: An Obj_Initial object for initializing raw Nanjing.
        """

        # factory inner Nanjing in gaming
        self.materials: list[Material] = []
        self.producers: list[Producer] = []
        self.price_source: Dict[datetime, Union[float]] = {}

        # origin Nanjing
        self.raw = {
            "material": list[Material],
            "producer": list[Producer],
        }
        # other Nanjing that will be used in the environment
        # connect mySQL database
        _database_info = _load_database_info()

        # database start date
        self.date_start: datetime = datetime(2022, 2, 1)
        self.date: datetime = self.date_start

        # pass database to obj_initial get the raw Nanjing of material and producer
        self.obj_ini = Obj_Initial(DB_type="CSV")
        # get the raw Nanjing
        self.raw["material"] = self.obj_ini.material_initialize()
        self.raw["producer"] = self.obj_ini.producer_initialize()
        self.price_source = self.obj_ini.price_initialize()
        # initialize
        self.reset(0)

    def reset(self, day_plus: int = 0) -> None:
        """
        Reset the factory by restoring materials and producers to their initial state.
        """
        self.materials = self.raw["material"]
        self.producers = self.raw["producer"]
        self.date = self.date_start + timedelta(day_plus)

    def info(self) -> (torch.Tensor, list, int):
        """
        Get information about the factory's materials and producers.

        :returns: A tuple containing the environment tensor and matrix size.
        :rtype: tuple[torch.tensor, list[int]]
        """
        mat_info = []
        pro_info = []
        # generate the material Nanjing matrix
        for item in self.materials:

            item.update_price(self.date, self.price_source[item.un_id])

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

        # generate the producer Nanjing matrix
        for item in self.producers:

            for mat_key, mat_value in item.material.items():
                pro_info.append([
                    int(item.un_id),
                    item.daily_low_cost,
                    item.daily_produce_cap,
                    int(mat_key),
                    mat_value,
                ])

        mat_count = len(mat_info)
        pro_count = len(pro_info)
        mat_colum = len(mat_info[-1])
        pro_colum = len(pro_info[-1])
        matrix_size = [1, mat_colum + pro_colum+1, mat_count if mat_count > pro_count else pro_count]

        # transfer list matrix into tensor

        def write_tensor(target: torch.tensor, matrix: list[list], m_count: int, m_colum: int, start: list):
            for m_c in range(m_count):
                for m_l in range(m_colum):
                    target[m_l + start[0], m_c + start[1]] = matrix[m_c][m_l]

        env_tensor = torch.zeros(matrix_size, dtype=torch.float32)
        write_tensor(env_tensor[0], mat_info, mat_count, mat_colum, [0, 0])
        write_tensor(env_tensor[0], pro_info, pro_count, pro_colum, [mat_colum + 1, 0])

        num_actions = len(self.materials)+len(self.producers)
        return env_tensor, matrix_size, num_actions

    def step(self, action: list[float], mode: str = "train") -> Dict[str, torch.tensor]:  # make one step forward
        """
         Take one step forward in the factory environment.

         :param list[float] action: A list of actions to be performed.
         :param str mode: The mode in which the factory is running ("train" or "play"). Default is "train".

         :returns: A dictionary containing relevant information based on the chosen mode.
         :rtype: Dict[str, torch.tensor]

         This method simulates one step in the factory environment based on the provided actions.
         It takes actions for trading and producing, computes the results, and returns information
         such as earnings, rewards, and outputs. The returned information depends on the mode specified.
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
                self.materials[act].trade(trade_action[act], self.date, self.price_source)
            )

        # produce
        for act in range(pro_act_count):
            produce_result.append(
                self.producers[act].produce(produce_action[act], self.materials)
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
            total_output = trade_output + produce_output
            return {
                "total_earn": total_earn,
                "total_reward": total_reward,
                "total_output": total_output
            }

        # match dictionary

        if mode == "train":
            _result = train_return()
        elif mode == "play":
            _result = play_return()
        else:
            _result = {}
        self.date += timedelta(days=1)
        for i in self.materials:
            i.inventory_change("refresh")
        return _result


if __name__ == '__main__':
    # a demo to test info and step
    example = Factory()
    example.reset(6)
    _, _, act = example.info()
    print(act)

    print(
        example.step([66,0,0,0,0,0,0,0,0,0,0,0,0], "train")
    )

