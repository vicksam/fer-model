import csv
from itertools import islice
import os
import pandas as pd


UNIFIED_DATASET_FILE_NAME = 'dataset.csv'

DATASET_NAMES = {'Training'   : 'train',
                 'PublicTest' : 'valid',
                 'PrivateTest': 'test'}

COLUMN_NAMES = ['dataset', 'image', 'fer_code', 'neutral', 'happiness', \
'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', \
'no-face']


def get_dataset_dict(dataset_dir = '../dataset',
                     fer_file_name = 'fer2013.csv',
                     fer_plus_file_name = 'fer2013new.csv'):
    '''Reads the output data csv (creates it first if it doesn't exist) into a
    dict.

    Args:
        dataset_dir(string): a path to a directory with dataset files
        fer_file_name(string): a name of fer csv file
        fer_plus_file_name(string): a name of fer plus csv file

    Returns: a dictionary of three dataset dataframes ('train', 'valid', 'test').
    '''
    # Check if the output csv dataset exists
    dataset_path = os.path.join(dataset_dir, UNIFIED_DATASET_FILE_NAME)
    if os.path.isfile(dataset_path):
        dataset_df = read_dataset_csv(dataset_dir)
    else:
        dataset_df = _generate_dataset_csv(dataset_dir,
                                           fer_file_name,
                                           fer_plus_file_name)

    return {'train' : dataset_df.loc[dataset_df['dataset'] == 'train'],
            'valid' : dataset_df.loc[dataset_df['dataset'] == 'valid'],
            'test' : dataset_df.loc[dataset_df['dataset'] == 'test']}

def read_dataset_csv(dataset_dir = './'):
    '''Reads into a dataframe a previously generated output dataset csv file.

    Args:
        dataset_dir(string): a path to a directory with dataset files

    Returns: a dataframe containing output dataset.
    '''
    dataset_path = os.path.join(dataset_dir, UNIFIED_DATASET_FILE_NAME)
    return pd.read_csv(dataset_path)

def _generate_dataset_csv(dataset_dir = '../dataset',
                          fer_file_name = 'fer2013.csv',
                          fer_plus_file_name = 'fer2013new.csv'):
    '''Generates output dataset csv file out of original fer and fer plus files.
    Saves it in the dataset directory.

    Args:
        dataset_dir(string): a path to a directory with dataset files
        fer_file_name(string): a name of fer csv file
        fer_plus_file_name(string): a name of fer plus csv file

    Returns: a dataframe contatining output dataset.
    '''
    # File paths
    fer_path = os.path.join(dataset_dir, fer_file_name)
    ferplus_path = os.path.join(dataset_dir, fer_plus_file_name)
    dataset_path = os.path.join(dataset_dir, UNIFIED_DATASET_FILE_NAME)

    # Create writer
    output_file = open(dataset_path, 'w')
    writer = csv.DictWriter(output_file, fieldnames = COLUMN_NAMES)
    writer.writeheader()

    # Read ferplus csv
    ferplus_entries = []
    with open(ferplus_path, 'r') as csvfile:
        ferplus_rows = csv.reader(csvfile, delimiter = ',')
        for row in islice(ferplus_rows, 1, None):
            ferplus_entries.append(row)

    # While reading fer csv, write to the output dataset csv,
    # combining old data with new labels
    index = 0
    with open(fer_path,'r') as csvfile:
        fer_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(fer_rows, 1, None):
            ferplus_row = ferplus_entries[index]
            file_name = ferplus_row[1].strip()
            if len(file_name) > 0:
                # dataset type, image string, counts for each emotion
                new_row = {
                    'dataset' : DATASET_NAMES[row[2]],
                    'image' : str(row[1]),
                    'fer_code' : str(row[0]),
                    'neutral' :  int(ferplus_row[2]),
                    'happiness' :  int(ferplus_row[3]),
                    'surprise' :  int(ferplus_row[4]),
                    'sadness' :  int(ferplus_row[5]),
                    'anger' :  int(ferplus_row[6]),
                    'disgust' :  int(ferplus_row[7]),
                    'fear' :  int(ferplus_row[8]),
                    'contempt' :  int(ferplus_row[9]),
                    'unknown' :  int(ferplus_row[10]),
                    'no-face' :  int(ferplus_row[11])
                }
                writer.writerow(new_row)
            index += 1

    output_file.close()

    # Return dataframe out of created dataset
    return pd.read_csv(dataset_path)
