# python standard
# JSON support to load the database info
import json

# pytorch
import torch


def _load_database_info():
    with open('.database', 'r') as json_file:
        # Load the JSON data into a Python dictionary
        return json.load(json_file)


# components of the factory
from .object import Material, Producer, Obj_Initial
from .tool_data import SQL


class factory:
    def __init__(self, date_plus, date_period):

        # factory inner data in gaming
        self.materials = []
        self.producers = []

        # origin data
        self.raw = {
            "material": [],
            "producer": [],
        }
        # other data that will be used in the environment
        # connect mySQL database
        _database_info = _load_database_info()
        self.database = SQL(
            host=_database_info["host"],
            port=_database_info["port"],
            user=_database_info["user"],
            password=_database_info["password"],
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

    def info(self) -> torch.tensor:
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

        # transfer list matrix into tensor
        env_tensor = torch.zeros()
        mat_tensor = torch.tensor(mat_info)



    def step(self, action: list[float]) -> dict:  # make one step forward

        pass


if __name__ == '__main__':


    pass