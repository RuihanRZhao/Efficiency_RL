"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""

# utilities
from src.game.factory.tool_data.t_sql import SQL


class Producer(object):
    def __init__(self, element: dict):
        """
        Initialize a Producer object.

        :param element: Initial configuration data for the producer.
        """
        self.un_id = ""
        self.material_list = []
        self.material_amount = []
        self.daily_low_cost = 0
        self.daily_produce_cap = 0
        # raw data for reset the factory
        self.raw_data = element
        # do initialize
        self.initialize()

    def __repr__(self):
        """
        Return a string representation of the Producer object.

        :return: String representation of the object.
        """
        return (
            f"{self.name}[{self.un_id}]\n"
            f"Origin Inventory: {self.inventory}  |  Capability of Inventory: {self.inventory_cap}\n"
            f"Origin Cache: {self.cache}  |  Capability of Cache: {self.cache_cap}\n"
            f"Raw Database: {self.database}"
        )

    def initialize(self) -> bool:
        """
        Initialize the Producer object with default values.

        :return: True if initialization is successful, else False.
        """
        self.un_id = ""
        self.material_list = []
        self.material_amount = []
        self.daily_low_cost = 0
        self.daily_produce_cap = 0
        return True

    def reset(self) -> bool:
        """
        Reset the Producer's properties to their initial values.

        :return: True if reset is successful, else False.
        """
        return self.initialize()

    def produce(self) -> tuple:
        """
        Produce items using the Producer.

        :return: A tuple containing production results.
        """
        return None, None
