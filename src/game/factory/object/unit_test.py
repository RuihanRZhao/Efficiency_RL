import unittest
from unittest.mock import Mock
from Material import Material
from src.game.factory.tool_data.t_sql import SQL


class material_test(unittest.TestCase):
    def setUp(self):
        # Create a mock SQL instance to use as a database
        self.mock_database = Mock(spec=SQL)
        self.material_data = [

        ]
        self.material_price_data = [

        ]
        self.mock_database.get_table_by_name.side_effect = [
            self.material_data, self.material_price_data
        ]

    def test_initialization_with_database(self):
        # Test the initialization of Material with a mock database
        material = Material(114514, "Shit", self.mock_database, max_store=100, max_extra_store=50,
                            ori_storage=10)

        self.assertEqual(material.un_id, 114514)
        self.assertEqual(material.name, "Shit")
        self.assertEqual(material.inventory, 10)
        self.assertEqual(material.inventory_cap, 100)
        self.assertEqual(material.cache, 0)
        self.assertEqual(material.cache_cap, 50)
        self.assertEqual(material.database, self.mock_database)
        self.assertEqual(material.raw_data, self.material_data)
        self.assertEqual(material.raw_price, self.material_price_data)

    def test_initialization_without_database(self):
        # Test the initialization of Material without a database
        with self.assertRaises(Exception):  # Replace with the actual exception type
            material = Material(114514, "Shit", None, max_store=100, max_extra_store=50, ori_storage=10)

