# utilities
from src.game.factory.tool_data import SQL
from material import Material
from producer import Producer


class Objects_Initial:
    """Initializes objects from a database.

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
        if database is None:
            raise ValueError("Do not have target server to get initialization data.")
        self.database = database
        self.material_list = []
        self.producer_list = []

    def material_initialize(self) -> list[Material]:
        """Initialize materials from the database.

        Returns:
            list[Material]: List of Material objects.
        """
        for element in self.database.get_table_by_name("material"):
            self.material_list.append(
                Material(
                    element
                )
            )
        return self.material_list

    def producer_initialize(self) -> list[Producer]:
        """Initialize producers from the database.

        Returns:
            list[Producer]: List of Producer objects.
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


if __name__ == '__main__':  # for individual test
    A = SQL(host="localhost", user="root", password="1919810", port=114, database="Factory")
    B = Objects_Initial(A)
    result = B.producer_initialize()
    print(B.producer_list)
