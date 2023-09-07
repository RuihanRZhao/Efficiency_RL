# utilities
from datetime import datetime
from typing import Dict, Union

from src.game.factory.tool_data import SQL
from material import Material as Material
from producer import Producer


class Objects_Initial:
    """
    Initializes objects from a database.

    Args:
        database (SQL | None): The database to retrieve initialization data from.
            Defaults to None.

    Raises:
        ValueError: If `database` is None.

    Attributes:
        database (SQL | None): The database used for initialization data.
        material_list (list[Material]): List to store Material objects.
        producer_list (list[Producer]): List to store Producer objects.
    """

    def __init__(self, database: SQL | None = None):
        """
        Initialize Objects_Initial with a database.

        Args:
            database (SQL | None): The database to retrieve initialization data from.
                Defaults to None.

        Raises:
            ValueError: If `database` is None.

        Attributes:
            database (SQL | None): The database used for initialization data.
            material_list (list[Material]): A list to store Material objects.
            producer_list (list[Producer]): A list to store Producer objects.
        """
        if database is None:
            raise ValueError("Do not have target server to get initialization data.")
        self.database = database
        self.material_list: list[Material] = []
        self.producer_list: list[Producer] = []
        self.price_dict: Dict[str, Dict[datetime, Union[float]]] = {}

    def material_initialize(self) -> list[Material]:
        """
        Initialize Material objects from the database.

        This method retrieves data from the database and creates Material objects
        based on that data. Each Material object represents a material available in
        the factory environment.

        Returns:
            list[Material]: A list of Material objects representing materials initialized from the database.
        """
        for element in self.database.get_table_by_name("material"):
            self.material_list.append(
                Material(
                    element
                )
            )
        return self.material_list

    def producer_initialize(self) -> list[Producer]:
        """
        Initialize Producer objects from the database.

        This method retrieves data from the database and creates Producer objects
        based on that data. Each Producer object represents a producer available in
        the factory environment.

        Returns:
            list[Producer]: A list of Producer objects representing producers initialized from the database.
        """
        raw = []
        producer_list = []
        for element in self.database.get_table_by_name("producer"):
            _if_change = False
            for item in raw:
                if item["un_id"] == element["un_id"]:
                    item["material"][f"{element['Material_id']}"] = element['Material_amount']
                    _if_change = True
                    break
            if not _if_change:
                raw.append({
                    "un_id": element["un_id"],
                    "daily_low_cost": element["daily_low_cost"],
                    "daily_produce_cap": element["daily_produce_cap"],
                    "material": {
                        element['Material_id']: element['Material_amount']
                    },
                })

        for item in raw:
            self.producer_list.append(Producer(item))

        return self.producer_list

    def price_initialize(self) -> Dict[str, Dict[datetime, Union[float]]]:
        price_dict: Dict[str, Dict[datetime, Union[float]]] = {}
        for element in self.database.get_table_by_name("Price"):
            if element["un_id"] in price_dict:
                price_dict[element["un_id"]][element["date"]] = element["price"]
            else:
                price_dict[element["un_id"]] = {}
                price_dict[element["un_id"]][element["date"]] = element["price"]

        return price_dict


if __name__ == '__main__':  # for individual test
    A = SQL(host="localhost", user="root", password="114514", port=114, database="Factory")
    B = Objects_Initial(A)
    result = B.price_initialize()
    print(result)
