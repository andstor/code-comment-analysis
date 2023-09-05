import pandas as pd
import numpy as np

import numpy as np
import constants

# mk function
def load_numpy_from_stringified_list(stringified_list):
    return np.fromstring(stringified_list, dtype=float, sep=',')


def load_all_data():
    frames = []
    for path in constants.data_files:
        df = pd.read_csv(path, sep=',')
        frames.append(df)
    df = pd.concat(frames)

    # serialize embeddings
    for col in  constants.embedding_column_names:
        df[col] = df[col].apply(load_numpy_from_stringified_list)

    return df

