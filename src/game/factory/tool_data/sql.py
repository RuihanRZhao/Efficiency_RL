import pymysql

class SQL:
    def __init__(self, host, user, password, port, database):
        self.Database = pymysql.connect(
            host=host,
            user=user,
            passwd=password,
            port=port,
        )
        self.table_list =

    def choose_table(self, database, name):

        table_list = []
