"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""
# python standard
from datetime import datetime, timedelta
# utilities
from src.game.factory.tool_data.t_sql import SQL


class Material(object):
    """
    Represents a material used in a factory process.

    Attributes:
        un_id (int): Unique identifier for the material.
        name (str): The name of the material.
        inventory (int): Current inventory storage.
        inventory_cap (int): Maximum storage capacity for inventory.
        cache (int): Current cache storage.
        cache_cap (int): Maximum extra storage capacity in cache.
        raw_data (dict): Raw data for resetting the factory.
    """

    def __init__(self, element: dict | None = None):
        self.un_id = ""
        assert isinstance(self.un_id, Material)

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
        self.raw_data = element
        """
            the format of the raw data should be like:
                {
                    "un_id": ""
                    "name": ""
                    "inventory": 0
                    "inventory_cap": 0
                    "cache": 0
                    "cache_cap": 0
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

    def initialize(self):
        self.un_id = self.raw_data["un_id"]
        self.name = self.raw_data["name"]
        self.inventory = self.raw_data["inventory"]
        self.inventory_cap = self.raw_data["inventory_cap"]
        self.cache = self.raw_data["cache"]
        self.cache_cap = self.raw_data["cache_cap"]
        return True

    def reset(self):
        return self.initialize()

    def load_price(self, date: datetime, source: dict | None = None):
        """
        Load the price data for a specific date.
        """
        now_price = source[date]
        trend = self.Trend_Cal(now_price, source[date-timedelta(days=3)], 3)
        self.price = {
            "date": date,
            "price_now": now_price,
            "price_trend": trend,
        }
        return self.price

    def Trend_Cal(self, end, start, scale):
        return (end - start) / scale

    def trade(self, amount):
        result = {
            "Earn": 0,
            "Reward": 0,
        }
