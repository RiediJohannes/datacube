from __future__ import annotations

import copy
import itertools
from collections.abc import Sequence
from enum import Enum


# enum to define the axis of a three-dimensional coordinate system
class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


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

    # Removes a row or column with the given name from the dataset
    def drop_line(self, name: str) -> None:
        # remove a row if applicable
        if name in self.row_names:
            row = self.row_names.index(name)
            del self.row_names[row]
            del self._values[row]
        # remove a column if applicable
        elif name in self.col_names:
            col = self.col_names.index(name)
            del self.col_names[col]
            for row in self._values:
                del row[col]

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
    Y_LABEL_LENGTH: int = 14

    def __init__(self, *, x: list[str], y: list[str], z: list[str]):
        self.x_labels = x
        self.y_labels = y
        self.z_labels = z
        self._tables: list[DataTable] = []

    def cell(self, x: int, y: int, z: int, verbose: bool = False) -> float | tuple[float, tuple[str]]:
        for param in [x, y, z]:
            if not isinstance(param, int):
                raise Exception("Invalid argument! Dimension indexes must be integers")

        x_str = self.x_labels[x] if -1 < x < len(self.x_labels) else None
        y_str = self.y_labels[y] if -1 < y < len(self.y_labels) else None
        z_str = self.z_labels[z] if -1 < z < len(self.z_labels) else None

        if None in [x_str, y_str, z_str]:
            raise Exception("Invalid argument! One or more parameters exceeded the possible dimension range (0-len)")

        if verbose:
            # noinspection PyTypeChecker
            return self._compute_value(x_str, y_str, z_str), (x_str, y_str, z_str)
        else:
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
        for y_val in self.y_labels:
            for x_val in self.x_labels:
                results.append(self._compute_value(x_val, y_val, self.z_labels[z_level]))

        print(' ' * (self.Y_LABEL_LENGTH + 2) + f"++ z-level: {self.z_labels[z_level]} ++")

        # print a pretty representation of the X-Y face
        width = 0
        margin_left = 0
        for i, _ in enumerate(self.y_labels):
            # construct a row for every x value
            line = ''
            for x in range(len(self.x_labels)):
                line = line + ' | ' + "{:.1f}".format(results[x + len(self.x_labels) * i])

            width: int = len(line) + 1
            margin_left: int = (self.Y_LABEL_LENGTH + 2)
            row_label: str = self.y_labels[i].rjust(self.Y_LABEL_LENGTH)[:self.Y_LABEL_LENGTH]

            # print the data row preceded by a dashed line
            print(' ' * margin_left + '-' * width)
            print(row_label + ' ' + line + ' |')
        # print the closing dashes
        print(' ' * margin_left + '-' * width)
        # show labels for the columns
        print(' ' * (margin_left + 1) + ' '.join([label.center(6)[:6] for label in self.x_labels]))

        return results

    def slice(self, axis: Axis, line: int | str) -> DataCube:
        # checks if the line label is valid (if line is string) or if the given index is within the available bounds
        def _check_line_selection(existing_labels, requested_line: int | str) -> int:
            line_index: int | str = requested_line
            if isinstance(requested_line, str):
                try:
                    line_index = existing_labels.index(requested_line)
                except ValueError:
                    raise Exception(f'Given label {requested_line} is not part of the labels on the respective axis')

            if line_index < 0 or line_index >= len(existing_labels):
                raise Exception(f'Out of bounds error! Z-axis does not contain element with index {line_index}')

            return int(line_index)

        # dice the current data cube with a single line on the requested axis to get the respective slice
        match axis:
            case Axis.X:
                index = _check_line_selection(self.x_labels, line)
                return self.dice([self.x_labels[index]], self.y_labels, self.z_labels)
            case Axis.Y:
                index = _check_line_selection(self.y_labels, line)
                return self.dice(self.x_labels, [self.y_labels[index]], self.z_labels)
            case Axis.Z:
                index = _check_line_selection(self.z_labels, line)
                return self.dice(self.x_labels, self.y_labels, [self.z_labels[index]])

    def dice(self, x_labels: list[str], y_labels: list[str], z_labels: list[str]) -> DataCube:
        # checks if the given lists contain only existing axis labels
        def _check_if_subset(small_list: list[str], big_list: list[str]):
            if not set(small_list).issubset(set(big_list)):
                raise Exception(f'Selected x-Labels {str(small_list)} are not a subset of the existing labels {str(big_list)}')

        # filters a given data table to keep only rows/columns whose label is in the allowed_items list
        def _filter_data_table(table: DataTable, allowed_items, current_items):
            for item in current_items:
                if item not in allowed_items:
                    table.drop_line(item)

        _check_if_subset(x_labels, self.x_labels)
        _check_if_subset(y_labels, self.y_labels)
        _check_if_subset(z_labels, self.z_labels)

        dice: DataCube = DataCube(x=x_labels, y=y_labels, z=z_labels)
        for data_table in self._tables:
            cloned_table = copy.deepcopy(data_table)

            _filter_data_table(cloned_table, x_labels, self.x_labels)
            _filter_data_table(cloned_table, y_labels, self.y_labels)
            _filter_data_table(cloned_table, z_labels, self.z_labels)

            dice.add_data(cloned_table)

        return dice

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
