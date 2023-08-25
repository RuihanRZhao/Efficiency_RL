"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""

# utilities
from src.game.factory.tool_data.t_sql import SQL


class Producer(object):
    def __init__(self, un_id: int, low_cost: int=0, day_cap: int=0, ):

        assert isinstance(un_id, Material)
        self.un_id = un_id
        self.material_list = []
        self.material_amount = []
        self.daily_low_cost = low_cost
        self.daily_produce_cap = day_cap
        # raw data for reset the factory
        self.raw_data = []
        self.initialize()

    def __repr__(self):
        return (
            f"{self.name}[{un_id}]\n"
            f"Origin Inventory: {self.inventory}  |  Capability of Inventory: {self.inventory_cap}\n"
            f"Origin Cache: {self.cache}  |  Capability of Cache: {self.cache_cap}\n"
            f"Raw Database: {self.database}"
        )

    def initialize(self):
        self.raw_data = self.database.get_table_by_name("producer")

        return

    def _load_price(self):
        pass
