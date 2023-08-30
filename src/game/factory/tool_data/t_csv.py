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

    A CSV file object with functions to process data

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

    def Read(self) -> list:
        """
        Read the contents of the target file.
        :return: A 2D list with the same data and structure of the origin file.
        :rtype: list[list[int]]
        """
        try:
            # Open the file in read mode
            with open(self.filename, 'r', newline="") as file:
                # Use csv.reader to read the data and create a 2D list
                return list(csv.reader(file, delimiter=self.separator))
        except FileNotFoundError:
            # If the file doesn't exist, print an error message and return an empty list
            print(f"Not Found File: {self.filename}")
            return []

    def OverWrite(self, head: list | None, content: list | None) -> None:
        """
        Overwrite the target file with new head and content.

        :param list head: The header for the new table.
        :param list content: The content to be written.
        """
        # Open the file in write mode, which will overwrite existing content
        with open(self.filename, 'w', newline="") as file:
            writer = csv.writer(file)
            # Write the header row
            if head is not None:
                writer.writerow(head)
            # Write the content rows
            if content is not None:
                for i in content:
                    writer.writerow(i)
            # Print a success message
            print(f"Overwrite {self.filename} succeed.")

    def Append(self, content: list | None) -> None:
        """
        Append content to the CSV file.

        :param content: A list containing data to append to the CSV file. Can be None.
        :type content: list or None
        """
        # Open the file in append mode
        with open(self.filename, 'a', newline="") as file:
            writer = csv.writer(file)
            # Iterate through the content and write each row
            for i in content:
                writer.writerow(i)
            # Print a success message
            print(f"Append {self.filename} succeed.")
