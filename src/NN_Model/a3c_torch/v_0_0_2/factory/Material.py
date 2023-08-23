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
        reward, value = 0, 0
        if self.max_storage >= self.storage + amount and amount >= 0:
            value -= amount * DB.Get_Material_Price(self.name, day)
            self.Update_Material_Stock(amount)
            reward += FV.Cost_Do_Nothing*2
        else:
            # raise ValueError("More than stock ability.")
            value = 0
            reward -= FV.Cost_Do_Nothing*2
        return reward, value

    def sell(self, amount, day):
        reward, value = 0, 0
        if self.storage >= amount >= 0:
            value += amount * DB.Get_Material_Price(self.name, day)
            self.Update_Material_Stock(-amount)
            reward += FV.Cost_Do_Nothing*3
        else:
            # raise ValueError("More than stock left.")
            value = 0
            reward -= FV.Cost_Do_Nothing*3
        return reward, value

    def Check_Material_Stock(self):
        database = DB.Get_DB_Method()
        cursor = database.cursor()
        sql = "select * from Efficiency_RL.material where name = (%s)"
        value = (self.name,)
        cursor.execute(sql, value)
        return cursor.fetchone()

    def Update_Material_Stock(self, amount_change):
        database = DB.Get_DB_Method()
        cursor = database.cursor()
        sql = "update Efficiency_RL.material set storage = (%s) where name = (%s)"
        material_state_now = self.Check_Material_Stock()
        value = (
            amount_change + int(material_state_now[2]),
            self.name,)
        cursor.execute(sql, value)
        database.commit()
        self.storage += amount_change

        return self.Check_Material_Stock()
