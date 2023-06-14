import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy



def main():
    # Chapter 2: Initial exploration of the dataset.

    DATASET_PATH = Path('./SensorLogger/')
    RESULT_PATH = Path('./intermediate_datafiles/')
    RESULT_FNAME = 'chapter2_result.csv'

    # Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
    # instance per minute, and a fine-grained one with four instances per second.
    GRANULARITIES = [2000, 1000]

    # We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
    [path.mkdir(exist_ok=True, parents=True) for path in [RESULT_PATH]]

    DataViz = VisualizeDataset(__file__)
    print('Please wait, this will take a while to run!')

    datasets = []
    for milliseconds_per_instance in GRANULARITIES:
        print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

        # Create an initial dataset object with the base directory for our data and a granularity
        dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

        # Add the selected measurements to it.
        # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
        # and aggregate the values per timestep by averaging the values
        # dataset.add_numerical_dataset('health.csv', 'startDate', ['value'], 'avg', '')
        dataset.add_numerical_dataset('AccelerometerUncalibrated.csv', 'time', ['x','y','z'], 'avg', 'acc_')
        dataset.data_table = dataset.data_table.astype(float)
        # dataset.add_numerical_dataset('accelerometer_smartwatch.csv', 'timestamps', ['x','y','z'], 'avg', 'acc_watch_')

        # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
        # and aggregate the values per timestep by averaging the values
        dataset.add_numerical_dataset('GyroscopeUncalibrated.csv', 'time', ['x','y','z'], 'avg', 'gyr_')
        # dataset.add_numerical_dataset('gyroscope_smartwatch.csv', 'timestamps', ['x','y','z'], 'avg', 'gyr_watch_')

        # We add the heart rate (continuous numerical measurements) and aggregate by averaging again
        dataset.add_numerical_dataset('HeartRate.csv', 'time', ['bpm'], 'avg', 'hr_')


        dataset.add_numerical_dataset('Location.csv', 'time', ['speed'], 'avg', 'loc_')
        # We add the labels provided by the users. These are categorical events that might overlap. We add them
        # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
        # occurs within an interval).
        dataset.add_event_dataset('Annotation2.csv', 'starttime', 'endtime', 'label', 'binary')

        # We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging
        # dataset.add_numerical_dataset('light_phone.csv', 'timestamps', ['lux'], 'avg', 'light_phone_')

        # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
        # and aggregate the values per timestep by averaging the values
        dataset.add_numerical_dataset('MagnetometerUncalibrated.csv', 'time', ['x','y','z'], 'avg', 'mag_')
        # dataset.add_numerical_dataset('magnetometer_smartwatch.csv', 'timestamps', ['x','y','z'], 'avg', 'mag_watch_')

        # We add the pressure sensed by the phone (continuous numerical measurements) and aggregate by averaging again
        # dataset.add_numerical_dataset('pressure_phone.csv', 'timestamps', ['pressure'], 'avg', 'press_phone_')

        # Get the resulting pandas data table
        dataset = dataset.data_table
        print(dataset)

        # Plot the data

        # Boxplot
        # DataViz.plot_dataset_boxplot(dataset, ['acc_phone_x','acc_phone_y','acc_phone_z','acc_watch_x','acc_watch_y','acc_watch_z'])
        DataViz.plot_dataset_boxplot(dataset, ['acc_x','acc_y','acc_z'])
        # DataViz.plot_dataset_boxplot(dataset, ['value'])

        # Plot all data
        # DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'label'],
        #                             ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'],
        #                             ['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])
        DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_', 'mag_', 'loc_', 'label'],
                                    ['like', 'like', 'like', 'like', 'like', 'like'],
                                    ['line', 'line', 'line', 'line', 'line', 'points'])
        # DataViz.plot_dataset(dataset, ['value'])

        # And print a summary of the dataset.
        util.print_statistics(dataset)
        datasets.append(copy.deepcopy(dataset))

        # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
        # dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')

    print("Below here are the differences!!")
    if len(GRANULARITIES) > 1:
        # Make a table like the one shown in the book, comparing the two datasets produced.
        util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

    print(dataset.dtypes)

    # Select columns with 'float64' dtype
    float64_cols = list(dataset.select_dtypes(include='float64'))

    # The same code again calling the columns
    dataset[float64_cols] = dataset[float64_cols].astype('float32')
    print("Changing the types to float32!")
    print(dataset.dtypes)



    # Finally, store the last dataset we generated (250 ms).
    dataset.to_csv(RESULT_PATH / RESULT_FNAME)

    # Lastly, print a statement to know the code went through
    print('The code has run through successfully!')


if __name__ == "__main__":
    main()