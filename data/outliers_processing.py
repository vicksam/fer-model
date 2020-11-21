import numpy as np
import pandas as pd

def get_dataset_without_custom_outliers(dataset_df, df_column_names):
    '''Returns a copy of a dataframe with removed outliers.'''
    # Make a copy not to change the original df
    dataset_df = dataset_df.copy()

    ## Change outlier counts of 1 to 0
    # Count their number
    n_outliers = np.count_nonzero(dataset_df.iloc[:, 3:].values == 1)
    # Consider only column names referring to emotion names
    for name in df_column_names[3:]:
        # Change every count of 1 to 0
        dataset_df[name] = dataset_df[name].apply(lambda x: 0 if x == 1 else x)
    print('Changed {} outlier votes of 1 to 0'.format(n_outliers))

    # Filters used to identify observations with maximum count being 'unknown'
    # or 'no-face'
    filters = [
        ("Voted 'unknown'",
         dataset_df.iloc[:, 3:].max(1) == dataset_df['unknown']),
        ("Voted 'no-face'",
         dataset_df.iloc[:, 3:].max(1) == dataset_df['no-face'])
    ]

    ## Filter out observations matching these filters
    for filter in filters:
        # Filter out their indexes
        indexes = dataset_df[filter[1]].index
        # Drop these observations and print their number
        dataset_df.drop(indexes, inplace = True)
        print('{}: {} observations have been removed'.format(filter[0],
                                                             len(indexes)))
    # Drop columns 'unknown' and 'no-face'
    dataset_df = dataset_df.iloc[:, :-2]

    return dataset_df

def get_dataset_without_original_outliers(dataset_df,
                                          cross_entropy,
                                          df_column_names):
    '''Removes outliers from dataframe as in the original Fer-Plus repository at
    https://github.com/microsoft/FERPlus.

    Args:
        dataset_df(dataframe) input df,
        cross_entropy(boolean): whether to apply cross-entropy or
                                majority preprocessing
        df_column_names(list): columns names of input df

    Returns: a dataframe without outlier records and votes.
    '''
    preprocessed_list = []

    for _, row in dataset_df.iterrows():
        orginal_votes = list(row[3:])
        processed_votes = _process_votes(orginal_votes, cross_entropy)
        index = np.argmax(processed_votes)
        if index < 8: # not unknown or no-face category
            processed_votes = processed_votes[:-2] # cut unknown and no-face

            # Concat original data with processed votes
            record = row[0:3].tolist() + processed_votes
            # Append
            preprocessed_list.append(record)

    preprocessed_df = pd.DataFrame(preprocessed_list,
                                   columns = df_column_names[:-2])
    return preprocessed_df

def _process_votes(original_votes, cross_entropy):
    '''Applies original outlier preprocessing of votes to the single record.

    Args:
        original_votes(list): list of integer vote counts for 10 labels
        cross_entropy(boolean): whether to apply cross-entropy or
                                majority preprocessing

    Returns: processed votes list for a single record.
    '''
    size = len(original_votes)
    votes_unknown = [0.0] * size
    votes_unknown[-2] = 1.0

    # Set vote counts of 1 to 0
    for i in range(size):
        if original_votes[i] < 1.0 + 0.01:
            original_votes[i] = 0.0

    sum_list = sum(original_votes)
    processed_votes = [0.0] * size

    if cross_entropy:
        sum_part = 0
        count = 0
        valid_votes = True
        while sum_part < 0.75*sum_list and count < 3 and valid_votes:
            maxval = max(original_votes) # peak value
            for i in range(size):
                if original_votes[i] == maxval:
                    processed_votes[i] = maxval # set peak value where there is peak value
                    original_votes[i] = 0 # empty the source
                    sum_part += processed_votes[i]
                    count += 1
                    if i >= 8:  # unknown or non-face share same number of max votes
                        valid_votes = False
                        # there have been other emotions ahead of unknown or non-face
                        if sum(processed_votes) > maxval:
                            processed_votes[i] = 0
                            count -= 1
                        break
        # less than 50% of the votes are integrated, or there are too many
        # emotions, we'd better discard this example
        if sum(processed_votes) <= 0.5*sum_list or count > 3:
            processed_votes = votes_unknown   # force setting as unknown
    # For majority - prediction either represents more than half of the votes
    # or is set to unknown
    else:
        # Find the peak value of the original_votes list
        maxval = max(original_votes)
        if maxval > 0.5*sum_list:
            processed_votes[np.argmax(original_votes)] = maxval
        else:
            processed_votes = votes_unknown   # force set votes to unknown

    return processed_votes
