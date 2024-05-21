# -*- coding: utf-8 -*-
# {
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

# }

# function to load and process the data

import pandas as pd

def load_and_preprocess_data(data_path):
    
    # Import the data from 'credit.csv'
    df = pd.read_csv(data_path)

    # Impute all missing values in all the features
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Drop 'Loan_ID' variable from the data
    df = df.drop('Loan_ID', axis=1)

    return df