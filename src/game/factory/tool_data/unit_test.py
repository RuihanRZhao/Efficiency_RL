import unittest as test
from unittest.mock import Mock
# test target
from .t_csv import CSV
from .t_sql import SQL
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


    def test_get_tables(self):
        sql = SQL("localhost", "root", "114514", 114, "Efficiency_RL_Ori")

        mock_table_list = [("material",), ("material_price",), ("producer",)]

        table_list = sql.get_tables()
        print(type(table_list))

        self.mock_cursor.fetchall.return_value = mock_table_list

        table_list = sql.get_tables()
        print(table_list)
        self.assertEqual(table_list, self.mock_cursor.fetchall.return_value)

    def test_get_table_by_name(self):
        sql = SQL("localhost", "root", "114514", 114, "Efficiency_RL_Ori")

        table_data = sql.get_table_by_name("material")

        print(table_data)
