"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""
# python standard
from datetime import datetime, timedelta
from typing import Dict, Union

class Material(object):
    def __init__(self, element: Dict[str, Union[str, int]]):
        """
        Initialize a Material object.

        Args:
            element (dict, optional): A dictionary containing initial values for the Material's properties.
        """
        self.un_id = ""

        # initialize all parameters of a material and default to "" or 0
        self.name = ""
        self.inventory = 0
        self.inventory_cap = 0
        self.cache = 0
        self.cache_cap = 0
        self.price = {
            "date": 0,
            "price_now": 0,
            "price_trend": 0.0,
        }
        # the storage of the original data of Material
        self.raw_data = element if element is not None else {
            "un_id": "",
            "name": "",
            "inventory": 0,
            "inventory_cap": 0,
            "cache": 0,
            "cache_cap": 0,
        }
        """
            the format of the raw data should be like:
                {
                    "un_id": "",
                    "name": "",
                    "inventory": 0,
                    "inventory_cap": 0,
                    "cache": 0,
                    "cache_cap": 0,
                }
        """
        self.initialize()

    def __repr__(self):
        """
        Return a string representation of the Material.

        Returns:
            str: A formatted string describing the Material's properties.
        """
        return (
            f"{self.name}[{self.un_id}]\n"
            f"Origin Inventory: {self.inventory}  |  Capability of Inventory: {self.inventory_cap}\n"
            f"Origin Cache: {self.cache}  |  Capability of Cache: {self.cache_cap}\n"
        )

    def initialize(self) -> bool:
        """
        Initialize the Material's properties based on the raw_data dictionary.

        Returns:
            bool: True if initialization is successful.
        """
        self.un_id = self.raw_data["un_id"]
        self.name = self.raw_data["name"]
        self.inventory = self.raw_data["inventory"]
        self.inventory_cap = self.raw_data["inventory_cap"]
        self.cache = self.raw_data["cache"]
        self.cache_cap = self.raw_data["cache_cap"]
        return True

    def reset(self) -> bool:
        """
        Reset the Material's properties to their initial values.

        Returns:
            bool: True if reset is successful.
        """
        return self.initialize()

    def load_price(self, date: datetime, source: Dict[datetime, float]) -> Dict[str, Union[datetime, float]]:
        """
        Load the price data for a specific date.

        Args:
            date (datetime): The date for which to load the price data.
            source (dict): A dictionary containing price data for different dates.

        Returns:
            dict: A dictionary containing the loaded price data.
        """
        now_price = source[date]
        trend = self.Trend_Cal(date, source, 3)
        self.price = {
            "date": date,
            "price_now": now_price,
            "price_trend": trend,
        }
        return self.price

    @staticmethod
    def Trend_Cal(end: datetime, price_source: Dict[datetime, float], scale: int) -> float:
        """
        Calculate the trend based on start and end values and a scaling factor.

        Args:
            end (datetime): The end value.
            price_source (dict): The table of price_data.
            scale (int): The scaling factor.

        Returns:
            float: The calculated trend value.
        """
        trend = 0.
        if end - timedelta(days=scale) in price_source:
            trend = (price_source[end] - price_source[end - timedelta(days=scale)])

        return trend

    def inventory_change(self, mode: str, amount: int = 0) -> Dict[str, Union[str, bool]]:
        """
        Change the inventory based on the given amount.

        This method allows you to perform different inventory changes depending on the specified mode.

        Args:
            mode (str): The mode of inventory change. Valid modes are: 'trade', 'produce', and 'refresh'.
            amount (int, optional): The amount by which to change the inventory. Defaults to 0.

        Returns:
            dict: A dictionary indicating whether the inventory was changed. The dictionary contains the following keys:
                - 'mode' (str): The input mode.
                - 'if_changed' (bool): A boolean indicating if the inventory was changed.
                - 'change_type' (str): A string indicating the type of change performed.

        Inventory Change Modes:
        - 'trade': Adjusts the inventory based on the trade amount. If the trade is successful, the inventory is updated,
                    and the 'if_changed' flag is set to True. Otherwise, the flag is set to False.
        - 'produce': Changes both inventory and cache based on the given amount. It first adds to the cache, and if
                     there's still an amount remaining, it adds to the inventory. If cache or inventory capacities are
                     reached, the remaining amount is not added.
        - 'refresh': Transfers cache to inventory if there's enough space in the inventory. If there's not enough space
                     in the inventory, the remaining cache amount is left in the cache.

        Note:
        - Positive 'amount' values represent addition to inventory or cache.
        - Zero 'amount' values indicate holding the current inventory state.
        - Negative 'amount' values represent reduction from inventory or cache.

        Example usage:
        ```python
        # Create a Material instance
        material = Material(element)

        # Change inventory in trade mode with a specified amount
        trade_result = material.inventory_change("trade", 100)

        # Change inventory in produce mode with a specified amount
        produce_result = material.inventory_change("produce", 50)

        # Change inventory in refresh mode
        refresh_result = material.inventory_change("refresh")
        ```

        """
        state = {
            "mode": mode,
            "if_changed": False,
            "change_type": ""
        }

        def cache_to_inventory() -> dict:
            change_state = {
                "if_changed": False,
                "change_type": "refresh"
            }
            space = self.inventory_cap - self.inventory
            if space >= self.cache:
                self.inventory += self.cache
                self.cache = 0
            elif space < self.cache:
                self.inventory = self.inventory_cap
                self.cache -= space
            return change_state

        def change_only_in_inventory(_amount: int = 0) -> dict:
            change_state = {
                "if_changed": False,
                "change_type": ""
            }

            if _amount > 0:
                change_state["change_type"] = "add"
                space = self.inventory_cap - self.inventory
                if space >= _amount:
                    self.inventory += _amount
                    _amount = 0
                    change_state["if_change"] = True
                else:
                    change_state["if_change"] = False

            elif _amount == 0:
                change_state["change_type"] = "hold"
                change_state["if_change"] = True

            elif _amount < 0:
                change_state["change_type"] = "reduce"
                if self.inventory >= _amount:
                    self.inventory -= _amount
                    _amount = 0
                    change_state["if_change"] = True
                else:
                    change_state["if_change"] = False

            return change_state

        def change_in_both(_amount: int = 0) -> dict:
            change_state = {
                "if_changed": False,
                "change_type": ""
            }
            if _amount > 0:
                change_state["change_type"] = "add"
                cache_space = self.cache_cap - self.cache
                # increase in cache first
                if cache_space >= _amount:
                    self.cache += _amount
                    _amount = 0
                    change_state["if_change"] = True
                else:
                    self.cache = self.cache_cap
                    _amount -= cache_space
                # rest go to inventory
                inventory_space = self.inventory_cap - self.inventory
                if inventory_space >= _amount:
                    self.inventory += _amount
                    _amount = 0
                    change_state["if_change"] = True
                else:
                    self.inventory = self.inventory_cap
                    _amount -= inventory_space
                    change_state["if_change"] = True

            elif _amount == 0:
                change_state["change_type"] = "hold"
                change_state["if_change"] = True

            elif _amount < 0:
                change_state["change_type"] = "reduce"

                if abs(_amount) < self.inventory + self.cache:
                    if self.inventory >= abs(_amount):
                        self.inventory += _amount
                        _amount = 0
                        change_state["if_change"] = True
                    else:
                        _amount += self.inventory
                        self.inventory = 0
                        self.cache += _amount
                        _amount = 0
                        change_state["if_change"] = True

            return change_state

        mode_list = {
            "trade": change_only_in_inventory(),
            "produce": change_in_both(),
            "refresh": cache_to_inventory()
        }

        if mode in mode_list:
            result = mode_list[mode](amount)
            state["if_changed"] = result["if_changed"]
            state["change_type"] = result["change_type"]

        return state

    def trade(self, amount: int, date: datetime, price_source: dict) -> Dict[str, Union[float, int, str]]:
        """
        Perform a trade action based on the given amount, date, and price data.

        Args:
            amount (int): The amount of the trade. Positive values represent buying, negative values represent selling,
                          and zero represents holding.
            date (datetime): The date of the trade.
            price_source (dict): A dictionary containing price data for different dates, where keys are datetime objects
                                and values are corresponding price values.

        Returns:
            dict: A dictionary containing trade-related information. The dictionary includes the following keys:
                - 'Earn' (float): The amount earned or lost from the trade, calculated as amount * price.
                - 'Reward' (int): The reward points earned or lost based on the success of the trade. Positive values
                                 indicate earning rewards, and negative values indicate losing rewards.
                - 'Output' (str): A descriptive string indicating the action taken in the trade, along with relevant
                                 details such as the amount, material name, trade type (buy/sell/hold), and price.

        Trade Decision Logic:
        - The function determines the trade decision (buy/sell/hold) based on the sign of the 'amount':
            - If 'amount' is positive, a buy action is recommended.
            - If 'amount' is zero, a hold action is recommended.
            - If 'amount' is negative, a sell action is recommended.
        - The function then checks whether the inventory_change operation for the specified amount was successful using
          the 'if_changed' flag.
        - If the inventory_change operation was successful:
            - The 'Earn' value is calculated by multiplying the 'amount' by the current price retrieved from 'price_source'.
            - The 'Output' string is updated to indicate the success of the trade action.
            - The 'Reward' is increased by 10 points to reflect the successful trade.
        - If the inventory_change operation failed:
            - The 'Output' string is updated to indicate the failure of the trade action.
            - The 'Reward' is decreased by 10 points to reflect the unsuccessful trade.

        Example usage:
        ```
        # Create a Material instance
        material = Material(element)

        # Define trade parameters
        trade_amount = 100
        trade_date = datetime(2023, 8, 25)
        trade_price_source = {...}  # A dictionary containing price data

        # Perform a trade action
        trade_result = material.trade(trade_amount, trade_date, trade_price_source)
        ```
        """

        result = {
            "Earn": 0,
            "Reward": 0,
            "Output": ""
        }

        price = self.load_price(date, price_source)["price_now"]
        if amount > 0:
            result["Output"] = f"Buy {amount} of {self.name}-{self.un_id} when price = {price}"
        elif amount == 0:
            result["Output"] = f"Hold {self.name}-{self.un_id} when price = {price}"
        elif amount < 0:
            result["Output"] = f"Sell {amount} of {self.name}-{self.un_id} when price = {price}"

        if self.inventory_change("trade", amount)["if_changed"]:
            result["Earn"] += amount * price
            result["Output"] += " succeed. "
            result["Reward"] += 10
        else:
            result["Output"] += " failed. "
            result["Reward"] -= 10

        return result

    def total_inventory(self):
        """
        Get the total inventory of the material.

        Returns:
            dict: A dictionary containing the total inventory and cache amounts.
        """
        return {
            "inventory": self.inventory,
            "cache": self.cache,
        }
