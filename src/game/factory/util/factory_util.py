from . import Database as DB

def Get_Material_by_Name(name, Material_List):
    for i in Material_List:
        if i.name == name:
            return i

def Get_producer_table():
    db = DB.Get_DB_Method()
    cursor = db.cursor()
    cursor.execute("select * from Efficiency_RL.producer")
    return cursor.fetchall()