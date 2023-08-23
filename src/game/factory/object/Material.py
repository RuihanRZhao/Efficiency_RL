"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""


class Material(object):
    def __init__(self, un_id, name, database: sql.SQL | None, max_store=0, max_extra_store=0, ori_storage=0):
        if database is None: raise ObjectError("Do not have target server to get initialization data.")

        assert isinstance(un_id, Material)
        self.un_id = un_id
        self.name = name if name is not None else ""
        self.inventory = ori_storage
        self.inventory_cap = max_store
        self.cache = 0
        self.cache_cap = max_extra_store
        self.price = []
        # raw data for reset the factory
        self.database = database
        self.raw_data = self.initialize()

    def __repr__(self):
        return ""

    def initialize(self, database):
        raw_data =

        return

    def _load_price(self):

