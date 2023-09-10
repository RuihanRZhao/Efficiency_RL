import pymysql
import pandas as pd

# MySQL database configuration
connection = pymysql.connect(
    host='localhost',
    port=114,
    user='root',
    password='114514',
    database='Factory',
)

csv_file = './price_T.csv'
df = pd.read_csv(csv_file)
try:
    with connection.cursor() as cursor:
        # Define the table name
        table_name = 'Price'

        for index, row in df.iterrows():
            sql = f"INSERT INTO {table_name} (un_id, date, price) VALUES (%s, %s, %s)"
            cursor.execute(sql, tuple(row))  # Replace column names and add placeholders accordingly
            print(tuple(row))

            connection.commit()

    # Commit the changes to the database

finally:
    # Close the database connection
    connection.close()

# INSERT INTO Factory.Price (`index`, un_id, date, price)
# VALUES (12310, '2022/2/1', '7726.39', null);
