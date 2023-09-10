import unittest
from datetime import datetime
from .material import Material
from .producer import Producer


class TestMaterial(unittest.TestCase):
    """
    NEED TO BE UPDATE
    """
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

    def test_unit(self, section: str, num: int, date: datetime, price: float, amount: int, result: bool, reward: int):
        print(f"Test: {section} - {num}: ", end="\n")
        mock_prices = {
            date: price,
        }
        earn = amount * mock_prices[date]
        inventory_after = self.mock_data["inventory"] - amount

        mock_prices = {date: price, }
        date_to_test = date
        result, action_type = self.material.trade(amount, date_to_test, mock_prices)
        self.assertEqual(result, (action_type[4:] == "succeed"))
        self.assertEqual(result["Earn"], earn)
        self.assertEqual(reward, result["Reward"])
        self.assertEqual(inventory_after, self.material.inventory)

    def test_trade_buy(self):
        self.material.reset()
        self.test_unit("Buy", 1, datetime(2023, 8, 23), 65, 20, True, 10)

    def test_trade_sell(self):
        self.material.reset()
        self.test_unit("Buy", 1, datetime(2023, 8, 23), 65, -30, True, 10)

    def test_trade_hold(self):
        self.material.reset()
        self.test_unit("Buy", 1, datetime(2023, 8, 23), 65, 0, True, 10)



class TestProducer(unittest.TestCase):

    def setUp(self):
        # Initialize any common setup code here.
        pass

    def tearDown(self):
        # Clean up after each test case if needed.
        pass

    def test_produce_valid(self):
        # Test producing goods with valid data.
        element = {
            "un_id": "123",
            "material": {
                "A": 5,
                "B": -3,
            },
            "daily_low_cost": 10.0,
            "daily_produce_cap": 20,
        }
        producer = Producer(element)

        materials = []  # Replace with actual Material objects
        result = producer.produce(15, materials)

        # Define expected results
        expected_result = {
            "Earn": 0,
            "Reward": 0,
        }

        # Perform assertions
        self.assertEqual(result, expected_result)

    def test_produce_invalid(self):
        # Test producing goods with invalid data.
        element = {
            "un_id": "456",
            "material": {
                "A": -5,
            },
            "daily_low_cost": 5.0,
            "daily_produce_cap": 10,
        }
        producer = Producer(element)

        materials = []  # Replace with actual Material objects
        result = producer.produce(12, materials)

        # Define expected results
        expected_result = {
            "Earn": 0,
            "Reward": -10,
        }

        # Perform assertions
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
