from . import Database as DB
from .Material import Material
from .Producer import Producer


def Initialize_Factory_DB():
    db = DB.Get_DB_Method()
    C = db.cursor()

    sql = [
        "drop table if exists Efficiency_RL.object",
        "drop table if exists Efficiency_RL.material_price",
        "drop table if exists Efficiency_RL.producer",

        "create table Efficiency_RL.object(name varchar(256),storage int,Max_Store int,Max_Extra_Store int) SELECT * FROM Efficiency_RL_Ori.material;",
        "create table Efficiency_RL.material_price(name varchar(256),day int,price float) SELECT * FROM Efficiency_RL_Ori.material_price;",
        "create table Efficiency_RL.producer(Produce varchar(256),Origin varchar(256),Origin_Volume float) SELECT * FROM Efficiency_RL_Ori.producer;"
    ]

    for i in sql:
        C.execute(i)

def Initialize_Material():
    material_list = []
    db = DB.Get_DB_Method()
    cursor = db.cursor()
    cursor.execute("select * from Efficiency_RL.material")
    for i in cursor.fetchall():
        material_list.append(Material(i[0], i[1], i[2], i[3], i[4]))

    return material_list

def Initialize_Producer():
    producer_list = []
    db = DB.Get_DB_Method()
    cursor = db.cursor()
    cursor.execute("select distinct Produce from Efficiency_RL.producer")
    target_list = cursor.fetchall()
    for target in target_list:
        cursor.execute("select Origin, Origin_Volume from Efficiency_RL.producer where Produce = (%s)", target)
        ori_list = cursor.fetchall()
        producer_list.append(Producer(target[0], ori_list, 0, 5))

    return producer_list
