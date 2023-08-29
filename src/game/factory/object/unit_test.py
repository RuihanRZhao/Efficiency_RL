import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from src.game.factory.tool_data.t_sql import SQL  # You might need to adjust the import path
from Material import Material


class TestMaterial(unittest.TestCase):

    def setUp(self):
        # Initialize any common variables here that might be needed for testing
        self.mock_data = {
            "un_id": "12345",
            "name": "Mock Material",
            "inventory": 100,
            "inventory_cap": 200,
            "cache": 50,
            "cache_cap": 100,
        }
        self.material = Material(self.mock_data)

    def test_initialize(self):
        self.assertEqual(self.material.name, self.mock_data["name"])
        self.assertEqual(self.material.inventory, self.mock_data["inventory"])
        self.assertEqual(self.material.inventory_cap, self.mock_data["inventory_cap"])
        self.assertEqual(self.material.cache, self.mock_data["cache"])
        self.assertEqual(self.material.cache_cap, self.mock_data["cache_cap"])

    def test_reset(self):
        self.material.name = "Changed Name"
        self.material.inventory = 150
        self.material.reset()
        self.assertEqual(self.material.name, self.mock_data["name"])
        self.assertEqual(self.material.inventory, self.mock_data["inventory"])

    def test_load_price(self):
        mock_prices = {
            datetime(2023, 8, 20): 50,
            datetime(2023, 8, 21): 55,
            datetime(2023, 8, 22): 60,
            datetime(2023, 8, 23): 65,
        }
        date_to_test = datetime(2023, 8, 23)
        loaded_price = self.material.load_price(date_to_test, mock_prices)
        self.assertEqual(loaded_price["price_now"], mock_prices[date_to_test])

    def test_inventory_change(self):
        self.assertTrue(self.material.inventory_change(50))
        self.assertEqual(self.material.inventory, self.mock_data["inventory"] + 50)
        self.assertFalse(self.material.inventory_change(-300))
        self.assertEqual(self.material.inventory, self.mock_data["inventory"] + 50)

    def test_trade_buy(self):

        self.material.reset()
        mock_prices = {
            datetime(2023, 8, 23): 65,
        }
        date_to_test = datetime(2023, 8, 23)
        result, action_type = self.material.trade(20, date_to_test, mock_prices)
        self.assertEqual(action_type, "buy succeed")
        self.assertEqual(result["Earn"], 20 * mock_prices[date_to_test])
        self.assertEqual(result["Reward"], 10)
        self.assertEqual(self.material.inventory, self.mock_data["inventory"] + 20)

    def test_trade_sell(self):

        self.material.reset()
        mock_prices = {
            datetime(2023, 8, 23): 65,
        }
        date_to_test = datetime(2023, 8, 23)
        result, action_type = self.material.trade(-30, date_to_test, mock_prices)
        self.assertEqual("sel succeed", action_type)
        self.assertEqual(10, result["Reward"])
        self.assertEqual(self.mock_data["inventory"]-30, self.material.inventory)

    def test_trade_hold(self):
        self.material.reset()
        mock_prices = {
            datetime(2023, 8, 23): 65,
        }
        date_to_test = datetime(2023, 8, 23)
        result, action_type = self.material.trade(0, date_to_test, mock_prices)
        self.assertEqual("hol succeed", action_type)
        self.assertEqual(10, result["Reward"])
        self.assertEqual(self.mock_data["inventory"], self.material.inventory)

if __name__ == '__main__':
    unittest.main()
