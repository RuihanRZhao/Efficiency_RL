"""
Object Class: object

all materials during the producing in target factory system,
including ingredient, intermediate and product



"""


class Material(object):
    def __init__(self, un_id, name, max_store=0, max_extra_store=0, ori_storage=0):
        assert isinstance(un_id, Material)
        self.un_id = un_id
        self.name = name if name is not None else ""
        self.inventory = ori_storage
        self.inventory_cap = max_store
        self.cache = 0
        self.cache_cap = max_extra_store
        self.price = []
        self._load_price()

    def __repr__(self):
        return (
            f"#################\n"
            f"type: Material\n"
            f"Unique ID: {self.un_id}\n"
            f"name: {self.name}\n"
            f"inventory: {self.inventory}\n"
            f"inventory capability: {self.inventory_cap}\n"
            f"cache: {self.cache}\n"
            f"max_cache: {self.cache_cap}\n"
        )

    def initialize(self):

        pass

    def _load_price(self):
        pass
