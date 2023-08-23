"""
Last Change: 2023/aug/23 11:28
Author: Ryen Zhao
"""
# utility packages
import pymysql


class SQL:
    """
    A utility class for managing SQL database connections and operations.

    This class provides a convenient way to establish a connection to an SQL database,
    retrieve a list of tables within a specified database, and execute various SQL queries.

    :param host: The hostname or IP address of the database server.
    :param user: The username for authenticating the database connection.
    :param password: The password for authenticating the database connection.
    :param port: The port number to use for the database connection.
    :param database: The name of the database to operate on.

    :ivar Database: The database connection object.
    :ivar cursor: The cursor object for executing SQL queries.
    :ivar table_list: A list of table names within the specified database.
    """

    def __init__(self, host, user, password, port, database):
        """
        Initialize an SQL object to manage database connections and operations.

        :param host: The hostname or IP address of the database server.
        :param user: The username for authenticating the database connection.
        :param password: The password for authenticating the database connection.
        :param port: The port number to use for the database connection.
        :param database: The name of the database to operate on.
        """
        self.DB_server = pymysql.connect(
            host=host,
            user=user,
            passwd=password,
            port=port,
        )
        self.database = database  # the name of target database in the server
        self.cursor = self.DB_server.Cursor()  # Initialize a cursor for database operations.
        self.table_list = self.get_tables()  # Fetch the list of tables in the specified database.

    def get_tables(self):
        """
        Retrieve a list of table names in the specified database.

        :return: A list of table names present in the specified database.
        """
        cursor.execute(f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='{self.database}'")
        result = cursor.fetchall()
        table_list = []
        for table in result: table_list.append(table[0])  # transfer data from tuple into a new list

        return table_list

    def get_table_by_name(self, table_name):
        """
                Fetch data from a specified table and return it as a list of dictionaries.

                :param table_name: The name of the table to fetch data from.
                :return: A list of dictionaries where each dictionary represents a row of data.
                """
        data: list[dict[Any, Any]] = []

        query = f"SELECT * FROM {table_name}"
        self.cursor.execute(query)

        columns = [_column[0] for _column in self.cursor.description]  # get column information

        for row in self.cursor.fetchall():
            data_dict = dict(zip(columns, row))
            data.append(data_dict)

        return data
