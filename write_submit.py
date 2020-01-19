import pandas as pd
import datetime

# Variable declaration
_PATH_TO_SUBMISSION_FILE = '/data/sample_submission.csv'
_NOW = str(datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))

# Load submission file
submit_df = pd.read_csv(_PATH_TO_SUBMISSION_FILE)

# Do logic HERE
submit_df['target'] = 10

# Write new submision file
_FILENAME = f'submits/sample_submission{_NOW}.csv'
print(f'Writing file: {_FILENAME}')
submit_df.to_csv(_FILENAME, header=True, index=False)




