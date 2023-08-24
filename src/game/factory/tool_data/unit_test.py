import unittest as test
from unittest.mock import Mock
# test target
from t_csv import CSV
from t_sql import SQL
# system utility
import os
import csv


class csv_test(test.TestCase):
    def setUp(self):
        with open("test.csv", mode="w"): pass

    def test_read_nonexistent_file(self):
        # Test reading a nonexistent file
        csv_object = CSV("nonexistent.csv", ",", "old")
        result = csv_object.Read()
        self.assertEqual(result, [])

    def test_read_existing_file(self):
        # Test reading data from an existing CSV file
        # Create a sample CSV file for testing
        sample_data = [["Name", "Age"], ["Alice", "25"], ["Bob", "30"]]
        with open("test.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(sample_data)

        csv_object = CSV("test.csv", ",", "old")
        result = csv_object.Read()
        self.assertEqual(result, sample_data)

    def test_overwrite_file(self):
        # Test overwriting a CSV file with new content
        sample_data = [["Name", "Age"], ["Alice", "25"], ["Bob", "30"]]
        csv_object = CSV("test.csv", ",", "new")
        csv_object.OverWrite(head=sample_data[0], content=sample_data[1:])

        with open("test.csv", "r", newline="") as file:
            reader = csv.reader(file)
            result = [row for row in reader]

        self.assertEqual(result, sample_data)

    def test_append_content(self):
        # Test appending content to an existing CSV file
        csv_object = CSV("test.csv", ",", "new")
        sample_data = [["Name", "Age"], ["Alice", "25"], ["Bob", "30"]]
        csv_object.OverWrite(head=sample_data[0], content=sample_data[1:])

        new_content = [["Carol", "28"], ["David", "40"]]
        csv_object.Append(content=new_content)

        with open("test.csv", "r", newline="") as file:
            reader = csv.reader(file)
            result = [row for row in reader]

        expected_result = sample_data + new_content
        self.assertEqual(result, expected_result)

    def tearDown(self):
        # Clean up by removing the test file
        try:
            os.remove("test.csv")
        except FileNotFoundError:
            pass


class sql_test(test.TestCase):
    def setUp(self):
        self.mock_cursor = Mock()
        self.mock_connection = Mock()
        self.mock_cursor.description = [("_column1",), ("_column2",)]  # Mock column description
        self.mock_cursor.fetchall.return_value = [(1, "data1"), (2, "data2")]  # Mock fetched data
        self.mock_connection.cursor.return_value = self.mock_cursor

        self.mock_pymysql = Mock()
        self.mock_pymysql.connect.return_value = self.mock_connection

        self.patcher = patch("your_module.pymysql", self.mock_pymysql)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_get_tables(self):
        sql = SQL("host", "user", "password", 3306, "test_db")

        mock_table_list = [("table1",), ("table2",)]
        self.mock_cursor.fetchall.return_value = mock_table_list

        table_list = sql.get_tables()

        self.assertEqual(table_list, ["table1", "table2"])

    def test_get_table_by_name(self):
        sql = SQL("host", "user", "password", 3306, "test_db")

        mock_table_data = [(1, "data1"), (2, "data2")]
        self.mock_cursor.fetchall.return_value = mock_table_data

        table_data = sql.get_table_by_name("test_table")

        self.assertEqual(table_data, [{"_column1": 1, "_column2": "data1"}, {"_column1": 2, "_column2": "data2"}])