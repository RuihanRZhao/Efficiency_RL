# python standard

# JSON support to load the database info
import json


def _load_database_info():
    with open('.database', 'r') as json_file:
        # Load the JSON data into a Python dictionary
        return json.load(json_file)


# components of the factory
from .object import Material, Producer, Obj_Initial
from .tool_data import SQL


class factory:
    def __init__(self):
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

    def step(self) -> dict:  # make one step forward
        pass


if __name__ == '__main__':


    pass