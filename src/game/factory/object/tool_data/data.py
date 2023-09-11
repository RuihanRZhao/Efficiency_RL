"""
Object Class: Data

Data 2023/Sept/10

Author: Ryen Zhao
"""
import os
from typing import List

from .t_sql import SQL
from .t_csv import CSV
from datetime import datetime


class Data:
    def __init__(self, data_source: str = "", data_type: str = ""):
        def _csv():
            material_table: list[dict] = CSV("Demo_data/Base/material.csv", ",", "old").Read()
            price_table: list[dict] = CSV("Demo_data/Base/material_price_D.csv", ",", "old").Read()
            producer_table: list[dict] = CSV("Demo_data/Base/producer_base.csv", ",", "old").Read()
            product_table: list[dict] = CSV("Demo_data/Base/producer.csv", ",", "old").Read()

            # type transfer
            for i in material_table:
                i["un_id"]: str = i["un_id"]
                i["name"]: str = i["name"]
                i["inventory"]: float = float(i["inventory"])
                i["inventory_cap"]: float = float(i["inventory_cap"])
                i["cache_cap"]: float = float(i["cache_cap"])
                i["purchase_permit"]: bool = True if i["purchase_permit"] else False
                i["sale_permit"]: bool = True if i["sale_permit"] else False

            for i in price_table:
                i["date"]: datetime = datetime.strptime(i["date"], "%Y-%m-%d")
                i["price"]: float = float(i["price"])

            for i in producer_table:
                i["daily_produce_cap"]: float = float(i["daily_produce_cap"])
                i["daily_low_cost"]: float = float(i["daily_low_cost"])

            for i in product_table:
                i["material_amount"]: float = float(i["material_amount"])

            return {
                "material": material_table,
                "price": price_table,
                "producer": producer_table,
                "product": product_table
            }

        def _sql(source: str = ""):
            return
        modes = {
            "csv": _csv(),
            "sql": _sql(),
        }
        self.data = modes[data_type]

    def get_table_by_name(self, name: str = "") -> list:
        return self.data[name]


if __name__ == "__main__":
    print(Data(data_type="csv").get_table_by_name("material"))
    print(Data(data_type="csv").get_table_by_name("price"))
    print(Data(data_type="csv").get_table_by_name("producer"))
    print(Data(data_type="csv").get_table_by_name("product"))



