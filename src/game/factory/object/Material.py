"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""

# utilities
from src.game.factory.tool_data.t_sql import SQL


class Material(object):
    """
    Represents a material used in a factory process.

    Args:
        name (str): The name of the material.
        un_id (int, optional): Unique identifier for the material. Default is None.
        database (SQL, optional): The database connection to retrieve initialization data.
        max_store (int, optional): Maximum storage capacity for inventory. Default is 0.
        max_extra_store (int, optional): Maximum extra storage capacity in cache. Default is 0.
        ori_storage (int, optional): Initial inventory storage. Default is 0.

    Attributes:
        un_id (int): Unique identifier for the material.
        name (str): The name of the material.
        inventory (int): Current inventory storage.
        inventory_cap (int): Maximum storage capacity for inventory.
        cache (int): Current cache storage.
        cache_cap (int): Maximum extra storage capacity in cache.
        database (SQL): The database connection for initialization data.
        raw_data (list[dict]): Raw data for resetting the factory.
        raw_price (list[dict]): Raw price data for the material.

    """

    def __init__(self, name: str, un_id: int | None = None, database: SQL | None = None,
                 max_store=0, max_extra_store=0, ori_storage=0):
        if database is None:
            raise ValueError("Do not have target server to get initialization data.")

        assert isinstance(un_id, Material)
        self.un_id = un_id
        self.name = name if name is not None else ""
        self.inventory = ori_storage
        self.inventory_cap = max_store
        self.cache = 0
        self.cache_cap = max_extra_store
        self.database = database
        self.raw_data = []
        self.raw_price = []
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
            f"Raw Database: {self.database}"
        )

    def raw_storage(self, data: list | None):
        """
        Initialize raw_data and raw_price based on database information.
        """
        self.raw_data = [(element if element["name"] is self.name else None) for element in
                         self.database.get_table_by_name("material")]
        self.raw_price = [(element if element["name"] is self.name else None) for element in
                          self.database.get_table_by_name("material_price")]

    def load_price(self, date):
        """
        Load the price data for a specific date.

        :param date: The date for which to load the price data.
        :type date: Any (add type here)
        :return: The price data for the specified date.
        :rtype: dict
        """
        return self.raw_price[date]

