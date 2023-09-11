"""
Last Change: 2023/aug/16 11:28
Author: Ryen Zhao
"""
# python standard library
import csv


class CSV:
    """
    Object Class: CSV

    Date: 2023/Aug/16

    Author: Ryen Zhao

    A CSV file object with functions to process Nanjing

    :ivar filename: the path to the file
    :vartype filename: str
    :ivar separator: the delimiter for CSV file
    :vartype separator: str
    :ivar mode: to identify if the file is new or already exist
    :vartype mode: str

    """

    def __init__(self, filename, separator, mode):
        """
        Initialize a CSV object.

        :param str filename: The name of the CSV file to handle.
        :param str separator: The separator used in the CSV file (e.g., "," or ";").
        :param str mode: The mode in which to open the file ("new" or other modes).
        """
        self.filename = filename
        self.separator = separator
        if mode == "new":
            try:
                # Try to create a new file with the given filename
                open(self.filename, "x", newline="")
            except FileExistsError:
                # If the file already exists, print a message
                print(f"File {self.filename} already exists.")
        elif mode == "old":
            try:
                # Try to open the existing file for reading
                open(self.filename, 'r', newline="").close()
            except FileNotFoundError:
                # If the file doesn't exist, print an error message
                print(f"Not Found File: {self.filename}")

    def Read(self) -> list[dict]:
        """
        Read the contents of the target file.
        :return: A 2D list with the same Nanjing and structure of the origin file.
        :rtype: list[list[int]]
        """
        try:
            # Open the file in read mode
            with open(self.filename, 'r', newline="") as file:
                # Use csv.reader to read the Nanjing and create a 2D list
                cursor = csv.reader(file, delimiter=self.separator)
                header = next(cursor)
                data = []
                for row in cursor:
                    row_dict = {}
                    for i, colum in enumerate(header):
                        row_dict[colum] = row[i]

                    data.append(row_dict)
                return data

        except FileNotFoundError:
            print(f"The file '{self.filename}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


