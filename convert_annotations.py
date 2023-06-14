import pandas as pd
import numpy as np


def main():
    # pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    directory = './SensorLogger/SensorLogger/'
    filename = 'Annotation.csv'
    # l = './HealthHeartRate_2023-06-160_11-01-39_SimpleHealthExportCSV/HealthHeartRate_2023-06-160_11-01-39_SimpleHealthExportCSV/HKQuantityTypeIdentifierHeartRateSeries_2023-06-160_11-01-42_SimpleHealthExportCSV.csv'
    df= pd.read_csv(directory + filename)
    print(df)
    # print(df.columns)
    df['duration'] = df['seconds_elapsed'].diff().shift(-1)
    df['starttime'] = df['time']
    # df['endtime'] = df['starttime'] + df['duration'] * 10**9
    df['endtime'] = df['starttime'].copy()
    df['endtime'].iloc[0:len(df) - 1] = df['starttime'].iloc[1:] - 1
    # df['endtime'] = df['endtime'].shift(-1).fillna(0).astype(int)
    # print(df['starttime'].shift(-1))
    # df['endtime'] = df['starttime'][1:]

    print(df)
    df.to_csv('Annotation.csv')
    # df['starttime'] = pd.to_datetime(df['starttime'])
    # df['starttime'] = df['starttime'].values.astype(np.int64)
    # df['endtime'] = pd.to_datetime(df['endtime'])
    # df['endtime'] = df['endtime'].values.astype(np.int64)
    # print(df)
    # df = df.drop(columns=['startDate'], axis=1)
    # print(df)
    # df.to_csv('Annotations.csv')

if __name__ == "__main__":
    main()