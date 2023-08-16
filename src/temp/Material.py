from . import Database as DB
from . import Fac_Value as FV


class Material(object):
    def __init__(self, id, name, storage=0, max_store=0, extra_store=0):
        self.id = id
        self.name = name
        self.storage = storage
        self.max_storage = max_store
        self.extra_storage = 0
        self.max_extra_storage = extra_store

    def __repr__(self):
        return "\n(object: Material, name: {}, storage: {}, max_storage: {}, extra_storage: {}, max_extra_Storage: {})".format(
            self.name, self.storage, self.max_storage, self.extra_storage, self.max_extra_storage)

    def buy(self, amount, day):
        reward = 0
        if self.max_storage >= self.storage + amount:
            reward -= amount * DB.Get_Material_Price(self.name, day)
        else:
            # raise ValueError("More than stock ability.")
            return FV.Cost_Do_Nothing

        self.Update_Material_Stock(amount)
        return reward

    def sell(self, amount, day):
        reward = 0
        if self.storage >= amount:
            reward += amount * DB.Get_Material_Price(self.name, day)
        else:
            # raise ValueError("More than stock left.")
            return FV.Cost_Do_Nothing


        self.Update_Material_Stock(-amount)
        return reward

    def Check_Material_Stock(self):
        database = DB.Get_DB_Method()
        cursor = database.cursor()
        sql = "select * from efficiency_rl.material where name = (%s)"
        value = (self.name,)
        cursor.execute(sql, value)
        return cursor.fetchone()

    def Update_Material_Stock(self, amount_change):
        database = DB.Get_DB_Method()
        cursor = database.cursor()
        sql = "update efficiency_rl.material set storage = (%s) where name = (%s)"
        material_state_now = self.Check_Material_Stock()
        value = (
            amount_change + int(material_state_now[2]),
            self.name,)
        cursor.execute(sql, value)
        database.commit()
        self.storage += amount_change

        return self.Check_Material_Stock()
