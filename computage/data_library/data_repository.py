"""
The module for getting datasets from the data repository of that project.
"""

import pandas as pd
import re
import gdown
import os


def import_data(file_name: str, url: str) -> pd.DataFrame(): # type: ignore
    """
    Downloads then parses file .txt with data from Illumina sequencing to dataframe

    :param file_name: name of sample data (.txt) in GoogleDrive
    :param url: link to file on disk
    :return: pd.DataFrame (rows = Feature_ID, columns = name of sample)
    """
    n = 30 #samples number
    if not os.path.exists(file_name):
        gdown.download(url, file_name, quiet=False, fuzzy=True)

    with open(file_name, "r") as file:
        # read ages
        for line in file:
            if re.match(r'^!Sample_characteristics_ch1.*age.*', line):
                ages = (line.split('\t')[1:])
                break

        for line in file:
            if re.match(r"^!series_matrix_table_begin", line):
                break

        df = []
        for line in file:
            df.append(line.split()[0:n])

        df = pd.DataFrame(df)
        df.columns = df.iloc[0]
        df.columns = df.columns.str.replace('"', '')
        df = df[1:]
        df = df.reset_index(drop=True)
        df.ID_REF = df.ID_REF.str.replace('"', '')
        samples = list(df.columns)[1:]
        df[samples] = df[samples].apply(pd.to_numeric)
        df.drop(df.tail(1).index, inplace=True)

        ages_items = [int(item.split(': ')[1].replace('"', '')) for item in ages]
        ages = ['ages']
        ages.extend(ages_items)

        ages = pd.DataFrame(ages[:n]).T
        ages.columns = df.columns
        df = pd.concat([df, ages], ignore_index=True)

    return df


# Example:
# SAMPLE_ID = 'GSE41169_series_matrix.txt'
# URL = 'https://drive.google.com/file/d/1cZca-FHnfeUerXJOzqhnH34mUYjNOgba/view?usp=sharing'
# data_test = import_data(SAMPLE_ID, URL)
# print(data_test)

