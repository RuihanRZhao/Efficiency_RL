# utilities
from src.game.factory.tool_data.t_sql import SQL
from .Material import Material
from .Producer import Producer


class Objects:
    def __init__(self, database: SQL | None = None):
        if database is None: raise ValueError("Do not have target server to get initialization data.")
        self.database = database
        self.material_list = []
        self.producer_list = []

    def material_initialize(self) -> list:
        for element in self.database.get_table_by_name("material"):
            self.material_list.append(
                Material(
                    element
                )
            )
        return self.material_list

    def producer_initialize(self):
        producer_list = []
        for element in self.database.get_table_by_name("producer"):
            if element["name"] in [i for i in producer_list]:
                pass
            else:
                pass
        return producer_list
