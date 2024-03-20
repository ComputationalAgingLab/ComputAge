"""
The module for getting datasets from the data repository of that project.
"""

from typing import Any
import pandas as pd
import re
import pandas as pd


def import_data(path_name: str) -> pd.DataFrame():
    """

    :param path_name: path to txt
    :return:
    """
    with open(path_name, "r") as file:
        for line in file:
            if re.match(r"^!series_matrix_table_begin", line):
                break

        df = []
        for line in file:
            df.append(line.split()[0:3])
        df = pd.DataFrame(df)
        df.drop(df.tail(1).index, inplace=True)

    return df

#example
# filename = "GSE41169_series_matrix.txt"

