"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""
# python standard
from typing import Dict, Union

from .material import Material

from src.game.factory.tool_data import SQL


class Producer(object):
    def __init__(self, element: Dict[str, Union[str, int, float, dict]]):
        """Initialize a Producer object.

        Args:
            element (Dict[str, Union[str, int, float, dict]]): Initial configuration data for the producer.
        """
        self.un_id = ""
        self.daily_low_cost = 0
        self.daily_produce_cap = 0
        self.material = {}
        # raw data for reset the factory
        self.raw_data = element
        # '''
        # Structure of raw_data
        # self.raw_date ={
        #     "un_id": 0,
        #     "material": {
        #         "A": 0.0,
        #     },
        #     "daily_low_cost": 0.0,
        #     "daily_produce_cap": 0,
        # }
        # '''

        # do initialize
        self.initialize()

    def __repr__(self):
        """Return a string representation of the Producer object.

        Returns:
            str: String representation of the object.
        """
        info = f"\n"
        info += f"Unique ID: {self.un_id}\n"
        info += f"Daily Low Cost: {self.daily_low_cost}\n"
        info += f"Daily Produce Cap: {self.daily_produce_cap}\n"
        info += f"Material Info:\n"
        for amt, mat in enumerate(self.material):
            info += f"\t{mat}: {amt}\n"

        return info

    def initialize(self) -> bool:
        """Initialize the Producer object with default values.

        Returns:
            bool: True if initialization is successful, else False.
        """
        self.un_id = self.raw_data["un_id"]
        self.material = self.raw_data["material"]
        self.daily_low_cost = self.raw_data["daily_low_cost"]
        self.daily_produce_cap = self.raw_data["daily_produce_cap"]
        return True

    def reset(self) -> bool:
        """Reset the Producer's properties to their initial values.

        Returns:
            bool: True if reset is successful, else False.
        """
        return self.initialize()

    def produce(self, amount: int, materials: list[Material]) -> dict:
        """Produce goods.

        Args:
            amount (int): The amount of goods to produce.
            materials (list[Material]): List of Material objects.

        Returns:
            dict: A dictionary containing information about the production result.
                The dictionary has keys "Earn" and "Reward" with corresponding values.
        """
        result = {
            "Earn": 0,
            "Reward": 0,
        }
        # check produce ability
        if amount > self.daily_produce_cap:
            result["Reward"] -= 10
        else:
            for element in materials:
                if element.un_id in self.material:
                    if self.material[element.un_id] < 0:
                        if not element.inventory + element.cache + self.material[element.un_id] > 0:
                            result["Reward"] -= 10
                    elif self.material[element.un_id] > 0:
                        if not element.inventory_cap + element.cache_cap - (
                                element.inventory + element.cache) > self.material[element.un_id]:
                            result["Reward"] -= 10

            if result["Reward"] > 0:
                for element in materials:
                    if element.un_id in self.material:
                        element.inventory_change("produce", amount)

        return result
