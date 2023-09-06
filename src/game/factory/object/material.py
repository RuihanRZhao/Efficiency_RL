"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""
# python standard
from datetime import datetime, timedelta
from typing import Dict, Union


class Material(object):
    """
    Represents a material within a factory environment.

    Args:
        element (dict, optional): A dictionary containing initial values for the Material's properties.
            Defaults to None.

    Attributes:
        un_id (str): A unique identifier for the material.
        name (str): The name of the material.
        inventory (int): The current quantity of the material in inventory.
        inventory_cap (int): The maximum capacity of the material's inventory.
        cache (int): The current quantity of the material in cache.
        cache_cap (int): The maximum capacity of the material's cache.
        trade_permit (dict): A dictionary indicating whether trading (purchase/sale) of this material is permitted.
        price (dict): A dictionary containing price-related information for the material.
        raw_data (dict): The storage of the original data of the Material.

    Methods:
        __init__(element=None):
            Initializes a Material object with optional initial properties.

        __repr__():
            Returns a string representation of the Material.

        initialize() -> bool:
            Initializes the Material's properties based on the raw_data dictionary.

        reset() -> bool:
            Resets the Material's properties to their initial values.

        load_price(date: datetime, source: dict) -> dict:
            Loads the price data for a specific date from a source dictionary.

        Trend_Cal(end: datetime, price_source: dict, scale: int) -> float:
            Calculates the trend based on start and end values and a scaling factor.

        inventory_change(mode: str, amount: int = 0) -> dict:
            Changes the inventory based on the specified mode and amount.

        trade(amount: int, date: datetime, price_source: dict) -> dict:
            Performs a trade action based on the given amount, date, and price data.

        total_inventory() -> dict:
            Returns a dictionary containing the total inventory and cache amounts of the material.

    Example Usage:
        # Create a Material instance
        material = Material(element)

        # Change inventory in trade mode with a specified amount
        trade_result = material.inventory_change("trade", 100)

        # Perform a trade action
        trade_result = material.trade(100, trade_date, trade_price_source)
    """
    def __init__(self, element: Dict[str, Union[str, int, bool]]):
        """
        Initialize a Material object.

        :param dict element: A dictionary containing initial values for the Material's properties.

        :ivar str un_id: Unique identifier for the material.
        :ivar str name: Name of the material.
        :ivar int inventory: Current inventory amount.
        :ivar int inventory_cap: Inventory capacity.
        :ivar int cache: Current cache amount.
        :ivar int cache_cap: Cache capacity.
        :ivar dict trade_permit: Dictionary indicating trade permits for purchase and sale.
        :ivar dict price: Dictionary containing price-related information.
        :ivar dict raw_data: The storage of the original data of Material.
        """
        self.un_id = ""

        # initialize all parameters of a material and default to "" or 0
        self.name = ""
        self.inventory = 0
        self.inventory_cap = 0
        self.cache = 0
        self.cache_cap = 0
        self.trade_permit = {
            "purchase": False,
            "sale": False,
        }
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
        # """
        #     the format of the raw data should be like:
        #         {
        #             "un_id": "",
        #             "name": "",
        #             "inventory": 0,
        #             "inventory_cap": 0,
        #             "cache": 0,
        #             "cache_cap": 0,
        #         }
        # """
        self.initialize()

    def __repr__(self):
        """
        Return a string representation of the Material.

        :returns: A formatted string describing the Material's properties.
        :rtype: str
        """
        return (
            f"{self.name}[{self.un_id}]\n"
            f"Origin Inventory: {self.inventory}  |  Capability of Inventory: {self.inventory_cap}\n"
            f"Origin Cache: {self.cache}  |  Capability of Cache: {self.cache_cap}\n"
        )

    def initialize(self) -> bool:
        """
        Initialize the Material's properties based on the raw_data dictionary.

        :returns: True if initialization is successful.
        :rtype: bool
        """
        self.un_id = self.raw_data["un_id"]
        self.name = self.raw_data["name"]
        self.inventory = self.raw_data["inventory"]
        self.inventory_cap = self.raw_data["inventory_cap"]
        self.cache = self.raw_data["cache"]
        self.cache_cap = self.raw_data["cache_cap"]
        self.trade_permit ={
            "purchase": self.raw_data["purchase_permit"],
            "sale": self.raw_data["sale_permit"],
        }
        return True

    def reset(self) -> bool:
        """
        Reset the Material's properties to their initial values.

        :returns: True if reset is successful.
        :rtype: bool
        """
        return self.initialize()

    def load_price(self, date: datetime, source: Dict[datetime, float]) -> Dict[str, Union[datetime, float]]:
        """
        Load the price data for a specific date.

        :param datetime date: The date for which to load the price data.
        :param dict source: A dictionary containing price data for different dates.

        :returns: A dictionary containing the loaded price data.
        :rtype: dict
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

        :param datetime end: The end value.
        :param dict price_source: The table of price_data.
        :param int scale: The scaling factor.

        :returns: The calculated trend value.
        :rtype: float
        """
        trend = 0.
        if end - timedelta(days=scale) in price_source:
            trend = (price_source[end] - price_source[end - timedelta(days=scale)])

        return trend

    def inventory_change(self, mode: str, amount: int = 0) -> Dict[str, Union[str, bool]]:
        """
        Change the inventory based on the given amount.

        :param str mode: The mode of inventory change ('trade', 'produce', or 'refresh').
        :param int amount: The amount by which to change the inventory. Defaults to 0.

        :returns: A dictionary indicating whether the inventory was changed.
        :rtype: dict

        Inventory Change Modes:
            - 'trade': Adjusts the inventory based on the trade amount.
            - 'produce': Changes both inventory and cache based on the given amount.
            - 'refresh': Transfers cache to inventory if there's enough space in the inventory.

        Note:
            - Positive 'amount' values represent addition to inventory or cache.
            - Zero 'amount' values indicate holding the current inventory state.
            - Negative 'amount' values represent reduction from inventory or cache.

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

        :param int amount: The amount of the trade.
        :param datetime date: The date of the trade.
        :param dict price_source: A dictionary containing price data for different dates.

        :returns: A dictionary containing trade-related information.
        :rtype: dict


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
        """

        result = {
            "Earn": 0,
            "Reward": 0,
            "Output": ""
        }
        permit = False

        price = self.load_price(date, price_source)["price_now"]
        if amount > 0:
            permit = self.trade_permit["purchase"]
            result["Output"] = f"Buy {amount} of {self.name}-{self.un_id} when price = {price}"
        elif amount == 0:
            permit = True
            result["Output"] = f"Hold {self.name}-{self.un_id} when price = {price}"
        elif amount < 0:
            permit = self.trade_permit["sale"]
            result["Output"] = f"Sell {amount} of {self.name}-{self.un_id} when price = {price}"

        if permit:
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

        :returns: A dictionary containing the total inventory and cache amounts.
        :rtype: dict
        """
        return {
            "inventory": self.inventory,
            "cache": self.cache,
        }
