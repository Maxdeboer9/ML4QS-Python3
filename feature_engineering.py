import sys
import copy
import pandas as pd
import time
from pathlib import Path
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter3_result_final.csv'
RESULT_FNAME = 'chapter4_result.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))



def main():
    print_flags()

    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)
    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = pd.Timedelta(dataset.index[1] - dataset.index[0]).microseconds / 1000
    # milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).seconds*1000

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    if FLAGS.mode == 'aggregation':
        # Chapter 4: Identifying aggregate attributes.

        # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
        window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(60000)/milliseconds_per_instance)]

         #please look in Chapter4 TemporalAbstraction.py to look for more aggregation methods or make your own.

        for ws in window_sizes:

            dataset = NumAbs.abstract_numerical(dataset, ['hr_bpm'], ws, 'mean')
            # dataset = NumAbs.abstract_numerical(dataset, ['hr_bpm'], ws, 'std')
            # dataset = NumAbs.abstract_numerical(dataset, ['hr_bpm'], ws, 'max')
            # dataset = NumAbs.abstract_numerical(dataset, ['hr_bpm'], ws, 'slope')

        # DataViz.plot_dataset(dataset, ['hr_bpm', 'hr_bpm_temp_mean', 'hr_bpm_temp_std', 'hr_bpm_temp_slope', 'label'], ['exact', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'points'])
        DataViz.plot_dataset(dataset, ['hr_bpm', 'hr_bpm_temp_mean', 'label'], ['exact', 'like', 'like'], ['line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))

    if FLAGS.mode == 'frequency':
        # Now we move to the frequency domain, with the same window size.

        fs = float(1000)/milliseconds_per_instance
        ws = int(float(10000)/milliseconds_per_instance)
        dataset = FreqAbs.abstract_frequency(dataset, ['hr_bpm'], ws, fs)
        print(dataset.columns)
        # Spectral analysis.
        DataViz.plot_dataset(dataset, ['hr_bpm_max_freq', 'hr_bpm_freq_weighted', 'hr_bpm_pse', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line','points'])
        print("--- %s seconds ---" % (time.time() - start_time))

    if FLAGS.mode == 'final':

        ws = int(float(0.5*60000)/milliseconds_per_instance)
        fs = float(1000)/milliseconds_per_instance

        selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]

        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'slope')
        # TODO: Add your own aggregation methods here

        DataViz.plot_dataset(dataset, ['acc_x','gyr_x','hr_bpm','loc_speed','mag_x','pca_1', 'label'], ['like'] * 7, ['line'] * 6 + ['points'])


        # CatAbs = CategoricalAbstraction()

        # dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03, ws * 2, 2)
        print(dataset.columns)

        periodic_predictor_cols = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','loc_speed','mag_x','mag_y','mag_z']



        dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)


        # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

        # The percentage of overlap we allow
        window_overlap = 0.9
        skip_points = int((1-window_overlap) * ws)
        dataset = dataset.iloc[::skip_points,:]


        dataset.to_csv(DATA_PATH / RESULT_FNAME)

        DataViz.plot_dataset(dataset,['acc_x','gyr_x','hr_bpm','loc_speed','mag_x','pca_1', 'label'], ['like'] * 7, ['line'] * 6 + ['points'])
        print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help= "Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggeregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ", choices=['aggregation', 'frequency', 'final'])



    FLAGS, unparsed = parser.parse_known_args()

    main()