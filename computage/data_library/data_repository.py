"""
The module for getting datasets from the data repository of that project.
"""

import pandas as pd
import re
import gdown


def import_data(file_name: str, url: str) -> pd.DataFrame(): # type: ignore
    """
    Downloads then parses file .txt with data from Illumina sequencing to dataframe

    :param file_name: name of sample data (.txt) in GoogleDrive
    :param url: link to file on disk
    :return: pd.DataFrame (rows = Feature_ID, columns = name of sample)
    """
    gdown.download(url, file_name, quiet=False, fuzzy=True)

    with open(file_name, "r") as file:
        for line in file:
            if re.match(r"^!series_matrix_table_begin", line):
                break

        df = []
        for line in file:
            df.append(line.split()[0:3])
        df = pd.DataFrame(df)
        df.drop(df.tail(1).index, inplace=True)

    return df


# Example:
# SAMPLE_ID = 'GSE41169_series_matrix.txt'
# URL = 'https://drive.google.com/file/d/1cZca-FHnfeUerXJOzqhnH34mUYjNOgba/view?usp=sharing'
# data_test = import_data(SAMPLE_ID, URL)