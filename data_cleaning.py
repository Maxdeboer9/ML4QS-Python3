from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
from Chapter3.ImputationMissingValues import ImputationMissingValues
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set up file names and locations.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter2_result.csv'
RESULT_FNAME = 'chapter3_result_outliers.csv'
LOF_CONSTANT = 5

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    print("FLAGS:", FLAGS)
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():

    print_flags()
    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)

    except IOError as e:
        print('File not found, try to run the preceding crowdsignals scripts first!')
        raise e
    print(dataset)
    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset(__file__)

    # Step 1: Let us see whether we have some outliers we would prefer to remove.

    # Determine the columns we want to experiment on.
    # outlier_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'hr_bpm', 'mag_x', 'mag_y', 'mag_z', 'loc_speed']
    outlier_columns = ['hr_bpm', 'loc_speed']
    # Create the outlier classes.
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()
    #chose one of the outlier methods: chauvenet, mixture, distance or LOF via the argument parser at the bottom of this page.

    if FLAGS.mode == 'chauvenet':

        # And investigate the approaches for all relevant attributes.
        for col in outlier_columns:

            print(f"Applying Chauvenet outlier criteria for column {col}")

            # And try out all different approaches. Note that we have done some optimization
            # of the parameter values for each of the approaches by visual inspection.
            dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
            DataViz.plot_binary_outliers(
                dataset, col, col + '_outlier')

    elif FLAGS.mode == 'mixture':

        for col in outlier_columns:

            print(f"Applying mixture model for column {col}")
            dataset = OutlierDistr.mixture_model(dataset, col)
            DataViz.plot_dataset(dataset, [
                                 col, col + '_mixture'], ['exact', 'exact'], ['line', 'points'])
            # This requires:
            # n_data_points * n_data_points * point_size =
            # 31839 * 31839 * 32 bits = ~4GB available memory

    elif FLAGS.mode == 'distance':
        for col in outlier_columns:
            try:
                dataset = OutlierDist.simple_distance_based(
                    dataset, [col], 'euclidean', FLAGS.dmin, FLAGS.fmin)
                DataViz.plot_binary_outliers(
                    dataset.copy(deep=True), col, col + '_simple_dist_outlier')
            except MemoryError as e:
                print(
                    'Not enough memory available for simple distance-based outlier detection...')
                print('Skipping.')

    elif FLAGS.mode == 'LOF':
        for col in outlier_columns:
            try:
                dataset = OutlierDist.local_outlier_factor(
                    dataset, [col], 'euclidean', FLAGS.K)
                DataViz.plot_dataset(dataset, [col, col + '_lof'], [
                                     'exact', 'exact'], ['line', 'points'])
            except MemoryError as e:
                print('Not enough memory available for lof...')
                print('Skipping.')


    elif FLAGS.mode == 'final':

        # We use Chauvenet's criterion for the final version and apply it to all but the label data...
        # for col in [c for c in dataset.columns if not 'label' in c]:

            # print(f'Measurement is now: {col}')
            # dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
            # dataset.loc[dataset[f'{col}_outlier'] == True, col] = np.nan
            # del dataset[col + '_outlier']
        print([c for c in dataset.columns if not 'label' in c])
        for col in [c for c in dataset.columns if not 'label' in c]:
            try:
                dataset = OutlierDist.local_outlier_factor(
                    dataset, [col], 'euclidean', FLAGS.K)
                DataViz.plot_dataset(dataset, [col, col + '_lof'], [
                                     'exact', 'exact'], ['line', 'points'])
                dataset.loc[dataset[col + '_lof'] > LOF_CONSTANT, col] = np.nan
                dataset = dataset.drop(col + '_lof', axis=1)
            except MemoryError as e:
                print('Not enough memory available for lof...')
                print('Skipping.')




    # for column in outlier_columns:
    #     if
    # dataset.to_csv(DATA_PATH / ('temp_' + RESULT_FNAME))
    # Replacing the outliers with NaN values so we can impute them later
    # if FLAGS.mode == 'distance':
    #     for column in outlier_columns:
    #         dataset.loc[dataset[column + '_simple_dist_outlier'] == True, column] = np.nan
    #         dataset = dataset.drop(column + '_simple_dist_outlier', axis=1)
    # if FLAGS.mode == 'LOF':
    #     for column in outlier_columns:
    #         dataset.loc[dataset[column + '_lof'] > LOF_CONSTANT, column] = np.nan
    #         dataset = dataset.drop(column + '_lof', axis=1)

    ImputeFunctions = ImputationMissingValues()
    dataset = ImputeFunctions.impute_interpolate(dataset, ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'hr_bpm', 'mag_x', 'mag_y', 'mag_z', 'loc_speed'])
    dataset.to_csv(DATA_PATH / RESULT_FNAME)

    # Plot all data
    # DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'label'],
    #                             ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'],
    #                             ['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

    DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_', 'mag_', 'loc_', 'label'],
                                ['like', 'like', 'like', 'like', 'like', 'like'],
                                ['line', 'line', 'line', 'line', 'line', 'points'])


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: LOF, distance, mixture, chauvenet or final \
                        'LOF' applies the Local Outlier Factor to a single variable \
                        'distance' applies a distance based outlier detection method to a single variable \
                        'mixture' applies a mixture model to detect outliers for a single variable\
                        'chauvenet' applies Chauvenet outlier detection method to a single variable \
                        'final' is used for the next chapter", choices=['LOF', 'distance', 'mixture', 'chauvenet', 'final'])

    parser.add_argument('--C', type=float, default=2,
                        help="Chauvenet: C parameter")

    parser.add_argument('--K', type=int, default=5,
                        help="Local Outlier Factor:  K is the number of neighboring points considered")

    parser.add_argument('--dmin', type=float, default=0.10,
                        help="Simple distance based:  dmin is ... ")

    parser.add_argument('--fmin', type=float, default=0.99,
                        help="Simple distance based:  fmin is ... ")

    FLAGS, unparsed = parser.parse_known_args()

    main()