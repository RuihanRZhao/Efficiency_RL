"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""
# python standard
from typing import Dict, Union

from .material import Material


class Producer(object):
    """
    Represents a producer within the factory environment.

    Args:
        element (Dict[str, Union[str, int, float, dict]]): Initial configuration Nanjing for the producer.

    Attributes:
        un_id (str): A unique identifier for the producer.
        daily_low_cost (float): The daily operating cost for the producer.
        daily_produce_cap (int): The daily production capacity of the producer.
        material (dict): A dictionary representing the materials required by the producer for production.
        raw_data (Dict[str, Union[str, int, float, dict]]): The raw configuration Nanjing used for initialization.

    Methods:
        __init__(self, element: Dict[str, Union[str, int, float, dict]]): Initializes a Producer object.
        __repr__(self): Returns a string representation of the Producer object.
        initialize(self) -> bool: Initializes the Producer object with default values.
        reset(self) -> bool: Resets the Producer's properties to their initial values.
        produce(self, amount: int, materials: list[Material]) -> Dict[str, Union[int, float, str]]: Produces goods
            and returns information about the production result.

    Example:
        # Create a Producer instance
        producer = Producer(element)

        # Produce goods using the producer
        production_result = producer.produce(100, materials)

        # Reset the producer's properties to their initial values
        producer.reset()
    """
    def __init__(self, element: Dict[str, Union[str, int, float, dict]]):
        """Initialize a Producer object.

        Args:
            element (Dict[str, Union[str, int, float, dict]]): Initial configuration Nanjing for the producer.
        """
        self.un_id = ""
        self.daily_low_cost = 0
        self.daily_produce_cap = 0
        self.material: dict = {}
        # raw Nanjing for reset the factory
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
        self.un_id = self.raw_data["producer_id"]
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

    def produce(self, amount: float, materials: list[Material], mode: str = "normal") -> Dict[str, Union[int, float, str]]:
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
            "Output": ""
        }
        # check produce ability
        if amount > self.daily_produce_cap:
            result["Reward"] -= 10
            result["Output"] += f"Exceed Production Capability: input = {amount}, cap for [{self.un_id}] = {self.daily_produce_cap}\n"
        elif amount == 0:
            result["Reward"] = 0
            result["Output"] += f"Stop Product: [{self.un_id}]"
        elif amount < 0:
            result["Reward"] -= 100
            result["Output"] += f"Cannot take negative value for Product"
        else:
            for element in materials:
                if element.un_id in self.material:
                    if self.material[element.un_id] < 0:
                        if not element.inventory + element.cache + self.material[element.un_id] > 0:
                            result["Reward"] -= 10
                            result["Output"] += f"Exceed Inventory Stock: Material: [{element.un_id}] | input = {amount}, stock = {element.inventory + element.cache}\n"

                    elif self.material[element.un_id] > 0:
                        if not element.inventory_cap + element.cache_cap - (
                                element.inventory + element.cache) > self.material[element.un_id]:
                            result["Reward"] -= 10
                            result["Output"] += f"Exceed Inventory Capability: Material: [{element.un_id}] | input = {amount}, space = {element.inventory_cap + element.cache_cap - (element.inventory + element.cache)}\n"

        if mode == "normal":
            if result["Reward"] > 0:
                for element in materials:
                    if element.un_id in self.material:
                        element.inventory_change("produce", amount)
                result["Output"] += f"Produce {amount} in [{self.un_id}] succeed."
        if mode == "mock":
            if result["Reward"] > 0:
                for element in materials:
                    if element.un_id in self.material:
                        element.inventory_change("mock-produce", amount)
                result["Output"] += f"Produce {amount} in [{self.un_id}] succeed."


        return result
