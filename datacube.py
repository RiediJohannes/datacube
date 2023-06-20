import itertools
from collections.abc import Sequence


def _is_collection(obj):
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


class DataTable:

    def __init__(self, rows: list[str], cols: list[str]):
        self.row_names = rows
        self.col_names = cols
        self._values: list[list[int]] = []

    @property
    def values(self) -> list[list[int]]:
        return self._values

    @values.setter
    def values(self, vals: list[list[int]]):
        # check dimensions of values
        if (actual_row := len(vals)) != (expected_row := len(self.row_names)):
            raise Exception(f"Invalid dimensions! Expected {expected_row} rows but got {actual_row}")

        if (actual_col := len(vals[0])) != (expected_col := len(self.col_names)):
            raise Exception(f"Invalid dimensions! Expected {expected_col} columns but got {actual_col}")

        for row in vals:
            if len(row) != actual_col:
                raise Exception("Invalid dimensions! All rows must be of the same length")

        self._values = vals

    def get_value(self, first: str, second: str):
        if first == second:
            raise Exception(f"Combinations of a value with itself are not illegal. The value {first} was given for both parameters")

        if not (self.__contains__(first) and self.__contains__(second)):
            raise Exception("Invalid arguments! Dataset does not contain a column and row with the given names")

        # find the corresponding row and column index for the given string arguments
        row = next(i for i, name in enumerate(self.row_names) if name == first or name == second)
        col = next(i for i, name in enumerate(self.col_names) if name == first or name == second)

        return self._values[row][col]

    def __contains__(self, obj: str) -> bool:
        all_names = self.row_names + self.col_names

        if _is_collection(obj):
            for item in obj:
                if item not in all_names:
                    return False  # if at least one item is not contained, return false
            return True
        else:
            return obj in all_names


class DataCube:

    def __init__(self, *, x: list[str], y: list[str], z: list[str]):
        self.x_labels = x
        self.y_labels = y
        self.z_labels = z
        self._tables: list[DataTable] = []

    def cell(self, x: int, y: int, z: int) -> float:
        for param in [x, y, z]:
            if not isinstance(param, int):
                raise Exception("Invalid argument! Dimension indexes must be integers")

        x_str = self.x_labels[x] if -1 < x < len(self.x_labels) else None
        y_str = self.y_labels[y] if -1 < y < len(self.y_labels) else None
        z_str = self.z_labels[z] if -1 < z < len(self.z_labels) else None

        if None in [x_str, y_str, z_str]:
            raise Exception("Invalid argument! One or more parameters exceeded the possible dimension range (0-len)")

        return self._compute_value(x_str, y_str, z_str)

    def add_data(self, table: DataTable):
        # check the type of the argument
        if not isinstance(table, DataTable):
            raise Exception("Invalid data format! Expected argument of type DataTable")

        all_naming_sets = [set(self.x_labels), set(self.y_labels), set(self.z_labels)]

        if not set(table.row_names) in all_naming_sets:
            raise Exception("Row names of DataTable do not match any axis definition of the cube")

        if not set(table.col_names) in [set(self.x_labels), set(self.y_labels), set(self.z_labels)]:
            raise Exception("Column names of DataTable do not match any axis definition of the cube")

        self._tables.append(table)

    def show_front(self, z_level: int = 0) -> list[float]:
        # calculate all values on one X-Y-face and save them to a list
        results = []
        for x_val in self.x_labels:
            for y_val in self.y_labels:
                results.append(self._compute_value(x_val, y_val, self.z_labels[z_level]))

        # print a pretty representation of the X-Y face
        offset = 0
        width = 0
        for _ in self.x_labels:
            line = ''
            for x in range(len(self.y_labels)):
                line = line + ' | ' + "{:.1f}".format(results[x + offset])

            width = len(line) + 1
            print(' ' + '-' * width)
            print(line + ' |')
            offset += 3
        print(' ' + '-' * width)

        return results

    def _compute_value(self, first: str, second: str, third: str) -> float:
        # get every possible unique combination of two values taken from the parameters first, second and third
        pairs = list(itertools.combinations([first, second, third], 2))
        results = []

        for pair in pairs:
            for table in self._tables:
                if pair in table:
                    results.append(table.get_value(pair[0], pair[1]))

        if len(results) != 3:
            raise Exception("Failed to compute value! Datasets do not contain a unifying cell for all three arguments")

        return round(sum(results) / 3, 1)

    # define a method to access properties by bracket indexing -> cube[x]
    def __getitem__(self, arg):
        match arg:
            case 'x':
                return self.x_labels
            case 'y':
                return self.y_labels
            case 'z':
                return self.z_labels
            case _:
                raise Exception("Invalid argument! Bracket indexing is only supported for the values x/y/z")

    # define a string representation
    def __str__(self) -> str:
        return f"""
            Datacube [{len(self.x_labels)}x{len(self.y_labels)}x{len(self.z_labels)}]:
            \tx: {self.x_labels}
            \ty: {self.y_labels}
            \tz: {self.z_labels}             
        """
