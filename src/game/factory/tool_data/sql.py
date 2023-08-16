import pymysql

class SQL:
    def __init__(self, host, user, password, port, database):
        self.Database = pymysql.connect(
            host=host,
            user=user,
            passwd=password,
            port=port,
        )
        self.cursor = self.Database.Cursor()
        self.table_list = get_tables(database)

    def get_tables(self, database):
        table_list = []
