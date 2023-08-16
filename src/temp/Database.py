import pymysql as mc

F_DB = mc.connect(
    host="127.0.0.1",
    user="root",
    passwd="114514",
    port=114
)


def Get_Material_Price(name, day):
    C = F_DB.cursor()
    sql = f"select price from Efficiency_RL.material_price where name = (%s) and day = (%s)";
    value = (name, day)
    C.execute(sql, value)
    return C.fetchone()[0]


def Get_DB_Method():
    return F_DB


def check_max_table_lenght():
    C = F_DB.cursor()
    C.execute("SELECT COUNT(*) FROM Efficiency_RL.material;")
    mL = C.fetchone()[0]
    C.execute("SELECT COUNT(*) FROM Efficiency_RL.producer;")
    pL = C.fetchone()[0]
    return mL if mL > pL else pL

import csv
# CSV version
def ReadFile(filename, file_type, separator):
    with open(filename, 'r') as file:
        if file_type == "csv":
            return list(csv.reader(file, delimiter=separator))


def WriteFile(filename, content):
    with open(filename, 'a', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(content)

