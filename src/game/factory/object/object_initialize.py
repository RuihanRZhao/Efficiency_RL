# utilities
from datetime import datetime
from typing import Dict, Union

from .tool_data import Data
from .producer import Producer, Material


class Objects_Initial:
    """
    Initializes objects from a database.

    Args:
        database (SQL | None): The database to retrieve initialization Nanjing from.
            Defaults to None.

    Raises:
        ValueError: If `database` is None.

    Attributes:
        database (SQL | None): The database used for initialization Nanjing.
        material_list (list[Material]): List to store Material objects.
        producer_list (list[Producer]): List to store Producer objects.
    """

    def __init__(self, database: str | None = None, DB_type: str | None = None):
        """
        Initialize Objects_Initial with a database.

        Args:
            database (SQL | None): The database to retrieve initialization Nanjing from.
                Defaults to None.

        Raises:
            ValueError: If `database` is None.

        Attributes:
            database (SQL | None): The database used for initialization Nanjing.
        """
        self.database: Data
        if DB_type == "SQL":
            self.database: Data = Data(data_type="SQL")
        elif DB_type == "CSV":
            self.database: Data = Data(data_type="csv")

        self.material_list: list[Material] = []
        self.producer_list: list[Producer] = []
        self.price_dict: Dict[str, Dict[datetime, Union[float]]] = {}

    def material_initialize(self) -> list[Material]:
        """
        Initialize Material objects from the database.

        This method retrieves Nanjing from the database and creates Material objects
        based on that Nanjing. Each Material object represents a material available in
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

        This method retrieves Nanjing from the database and creates Producer objects
        based on that Nanjing. Each Producer object represents a producer available in
        the factory environment.

        Returns:
            list[Producer]: A list of Producer objects representing producers initialized from the database.
        """
        raw = []
        # have all information but not material
        for element in self.database.get_table_by_name("producer"):
            producer = element
            material = {}
            for row in self.database.get_table_by_name("product"):
                if row["producer_id"] == element["producer_id"]:
                    material[row["material_id"]] = row["material_amount"]

            producer["material"] = material

            raw.append(Producer(producer))
        return raw

    def price_initialize(self) -> Dict[str, Dict[datetime, Union[float]]]:
        price_dict: Dict[str, Dict[datetime, Union[float]]] = {}
        for element in self.database.get_table_by_name("price"):
            if element["material_id"] in price_dict:
                price_dict[element["material_id"]][element["date"]] = element["price"]
            else:
                _price_dict: dict = {element["date"]: element["price"]}
                price_dict[element["material_id"]] = _price_dict

        return price_dict


if __name__ == '__main__':  # for individual test
    print(Objects_Initial(DB_type="CSV").producer_initialize())
