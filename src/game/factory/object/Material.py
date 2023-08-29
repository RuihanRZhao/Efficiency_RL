"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""
# python standard
from datetime import datetime, timedelta


class Material(object):
    def __init__(self, element: dict):
        """
        Initialize a Material object.

        :param element: A dictionary containing initial values for the Material's properties.
        :type element: dict, optional
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

        :return: A formatted string describing the Material's properties.
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
        """
        return self.initialize()

    def load_price(self, date: datetime, source: dict) -> dict:
        """
        Load the price data for a specific date.

        :param date: The date for which to load the price data.
        :type date: datetime
        :param source: A dictionary containing price data for different dates.
        :type source: dict
        :return: A dictionary containing the loaded price data.
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
    def Trend_Cal(end: datetime, price_source: dict, scale: int) -> float:
        """
        Calculate the trend based on start and end values and a scaling factor.

        :param end: The end value.
        :param price_source: The table of price_data.
        :param scale: The scaling factor.
        :return: The calculated trend value.
        """
        trend = 0.
        if end-timedelta(days=scale) in price_source:
            trend = (price_source[end]-price_source[end-timedelta(days=scale)])

        return trend

    def inventory_change(self, amount: float) -> bool:
        """
        Change the inventory based on the given amount.

        :param amount: The amount by which to change the inventory.
        :type amount: float
        :return: A boolean indicating whether the inventory was changed.
        :rtype: bool
        """
        _if_changed = False
        if amount > 0:
            if amount + self.inventory <= self.inventory_cap:
                self.inventory += amount
                _if_changed = True
        elif amount < 0:
            if abs(amount) < self.inventory + self.cache:
                if self.inventory >= abs(amount):
                    self.inventory += amount
                    amount = 0
                    _if_changed = True
                else:
                    amount += self.inventory
                    self.inventory = 0
                    self.cache += amount
                    amount = 0
                    _if_changed = True
        else:
            _if_changed = True
        return _if_changed

    def trade(self, amount: float, date: datetime, price_source: dict) -> (dict, str):
        """
        Perform a trade action.

        :param amount: The amount of the trade.
        :type amount: float
        :param date: The date of the trade.
        :type date: datetime
        :param price_source: A dictionary containing price data for different dates.
        :type price_source: dict
        :return: A dictionary containing the trade result and the action type.
        :rtype: dict, str
        """
        result = {
            "Earn": 0,
            "Reward": 0,
        }
        Action_Type = ""
        if amount > 0:
            Action_Type = "buy"
        elif amount < 0:
            Action_Type = "sel"
        else:
            Action_Type = "hol"
        if self.inventory_change(amount):
            result["Earn"] += amount * self.load_price(date, price_source)["price_now"]
            result["Reward"] += 10
            Action_Type += " succeed"
        else:
            result["Reward"] -= 10
            Action_Type += " failed"

        return result, Action_Type
