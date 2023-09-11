"""
Object Class: Data

Data 2023/Sept/10

Author: Ryen Zhao
"""
from t_sql import SQL
from t_csv import CSV


class Data:
    def __init__(self, data_source: str = "", data_type: str = ""):
        def _csv(source: str = ""):

            return

        def _sql(source: str = ""):

            return

        modes = {
            "csv": _csv(),
            "sql": _sql(),
        }

        self.data = modes[data_type](data_source)
