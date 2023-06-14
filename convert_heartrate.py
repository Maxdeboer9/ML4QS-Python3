import pandas as pd
import numpy as np


def main():
    directory = './Heartrate/Heartrate/'
    filename = 'HKQuantityTypeIdentifierHeartRateSeries_2023-06-162_10-41-27_SimpleHealthExportCSV.csv'
    # l = './HealthHeartRate_2023-06-160_11-01-39_SimpleHealthExportCSV/HealthHeartRate_2023-06-160_11-01-39_SimpleHealthExportCSV/HKQuantityTypeIdentifierHeartRateSeries_2023-06-160_11-01-42_SimpleHealthExportCSV.csv'
    df= pd.read_csv(directory + filename)
    print(df)
    print(df.columns)
    df['starttime'] = pd.to_datetime(df['startDate'])
    df['starttime'] = df['starttime'].values.astype(np.int64)
    df['endtime'] = pd.to_datetime(df['endDate'])
    df['endtime'] = df['endtime'].values.astype(np.int64)
    print(df)
    df = df.drop(columns=['startDate', 'endDate'], axis=1)
    print(df)
    df.to_csv('HeartRate.csv')

if __name__ == "__main__":
    main()