import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split

from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
from Chapter7.LearningAlgorithms import ClassificationAlgorithms
from Chapter7.LearningAlgorithms import RegressionAlgorithms
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
from Chapter7.FeatureSelection import FeatureSelectionClassification
from Chapter7.FeatureSelection import FeatureSelectionRegression
from util import util
from util.VisualizeDataset import VisualizeDataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter5_result.csv'
RESULT_FNAME = 'chapter7_classification_result.csv'
EXPORT_TREE_PATH = Path('./figures/crowdsignals_ch7_classification/')

# Next, we declare the parameters we'll use in the algorithms.
N_FORWARD_SELECTION = 50

def read_dataset():
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e
    dataset.index = pd.to_datetime(dataset.index)
    return dataset


def main():
    N_KCV_REPEATS = 10
    dataset = read_dataset()
    DataViz = VisualizeDataset(__file__)

    prepare = PrepareDatasetForLearning()

    train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=False)

    print('Training set length is: ', len(train_X.index))
    print('Test set length is: ', len(test_X.index))
    # Select subsets of the features that we will consider:

    basic_features = ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','hr_bpm','mag_x','mag_y','mag_z', 'loc_speed']
    pca_features = ['pca_1','pca_2','pca_3','pca_4']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]
    # freq_features = [name for name in dataset.columns if '_freq' in name]

    print('#basic features: ', len(basic_features))
    print('#PCA features: ', len(pca_features))
    print('#time features: ', len(time_features))
    print('#frequency features: ', len(freq_features))
    # cluster_features = ['cluster']
    cluster_features = [name for name in dataset.columns if '_cluster' in name]
    print('#cluster features: ', len(cluster_features))

    # features_after_chapter_3 = list(set().union(basic_features, pca_features))
    # features_after_chapter_4 = list(set().union(basic_features, pca_features, time_features, freq_features))
    features_after_chapter_5 = list(set().union(basic_features, pca_features, time_features, freq_features, cluster_features))
    # old_selected_features = ['loc_speed_freq_0.0_Hz_ws_20', 'loc_speed_temp_mean_ws_60', 'hr_bpm_temp_mean_ws_60', 'hr_bpm',
    #                      'acc_x_temp_std_ws_60', 'pca_1_temp_std_ws_60', 'gyr_z_freq_1.0_Hz_ws_20', 'acc_y_freq_0.7_Hz_ws_20',
    #                      'acc_z_freq_0.7_Hz_ws_20', 'gyr_y_freq_0.7_Hz_ws_20', 'gyr_x_max_freq', 'acc_z_freq_0.9_Hz_ws_20',
    #                      'gyr_x_freq_0.5_Hz_ws_20', 'acc_y_freq_0.5_Hz_ws_20', 'loc_speed_max_freq']

    fs = FeatureSelectionClassification()
    N_FORWARD_SELECTION = 15

    features, tree_features, tree_scores = fs.forward_selection(N_FORWARD_SELECTION,
                                                                    train_X[features_after_chapter_5],
                                                                    test_X[features_after_chapter_5],
                                                                    train_y,
                                                                    test_y,
                                                                    gridsearch=False)

    DataViz.plot_xy(x=[range(1, N_FORWARD_SELECTION+1)], y=[tree_scores],
                    xlabel='number of features', ylabel='accuracy', title='Decision Tree Features')

    features, forest_features, forest_scores = fs.forward_selection(N_FORWARD_SELECTION,
                                                                    train_X[features_after_chapter_5],
                                                                    test_X[features_after_chapter_5],
                                                                    train_y,
                                                                    test_y,
                                                                    gridsearch=False, forest=True)

    DataViz.plot_xy(x=[range(1, N_FORWARD_SELECTION+1)], y=[forest_scores],
                    xlabel='number of features', ylabel='accuracy', title='Random Forest Features')

    print("Decision Tree Features:")
    for feature, score in zip(tree_features, tree_scores):
        print(feature, score)
    print("Final score: {}".format(tree_scores[-1]))
    print("Random Forest Features:")
    for feature, score in zip(forest_features, forest_scores):
        print(feature, score)
    print("Final score: {}".format(forest_scores[-1]))


    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    start = time.time()

    # selected_features = ordered_features

    possible_feature_sets = [features_after_chapter_5, tree_features, forest_features]
    feature_names = ['Chapter 5', 'Decision Tree Features', 'Random Forest Features']
    total_class_test_y = np.array([])
    scores_over_all_algs = []
    for i in range(0, len(possible_feature_sets)):
        selected_train_X = train_X[possible_feature_sets[i]]
        selected_test_X = test_X[possible_feature_sets[i]]

        performance_tr_rf = 0
        performance_te_rf = 0
        total_class_test_y = np.array([])
        for repeat in range(0, N_KCV_REPEATS):
            print("Training RandomForest run {} / {} ... ".format(repeat, N_KCV_REPEATS, feature_names[i]))
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
                selected_train_X, train_y, selected_test_X, gridsearch=True
            )
            print(len(total_class_test_y), len(class_test_y))
            total_class_test_y = np.concatenate([total_class_test_y, class_test_y])

            performance_tr_rf += eval.accuracy(train_y, class_train_y)
            performance_te_rf += eval.accuracy(test_y, class_test_y)

        overall_performance_tr_rf = performance_tr_rf/N_KCV_REPEATS
        overall_performance_te_rf = performance_te_rf/N_KCV_REPEATS

        print(overall_performance_tr_rf)
        print(overall_performance_te_rf)

        print("Training Descision Tree run 1 / 1  featureset {}:".format(feature_names[i]))
        class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(
            selected_train_X, train_y, selected_test_X, gridsearch=True
        )

        performance_tr_dt = eval.accuracy(train_y, class_train_y)
        performance_te_dt = eval.accuracy(test_y, class_test_y)

        scores_with_sd = util.print_table_row_performances(feature_names[i], len(selected_train_X.index), len(selected_test_X.index), [
                                                                                                    (overall_performance_tr_rf, overall_performance_te_rf),
                                                                                                    (performance_tr_dt, performance_te_dt)])
        scores_over_all_algs.append(scores_with_sd)

    DataViz.plot_performances_classification(['RF', 'DT'], feature_names, scores_over_all_algs)

    # And we study two promising ones in more detail. First, let us consider the decision tree, which works best with the
    # selected features.

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.decision_tree(train_X[tree_features], train_y, test_X[tree_features],
                                                                                               gridsearch=True,
                                                                                               print_model_details=True, export_tree_path=EXPORT_TREE_PATH)

    test_cm = eval.confusion_matrix(test_y, class_test_y, class_train_prob_y.columns)

    DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)

    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
        train_X[forest_features], train_y, test_X[forest_features],
        gridsearch=True, print_model_details=True)

    total_test_y = list(test_y['class']) * (N_KCV_REPEATS + 1)
    total_class_test_y = np.concatenate([total_class_test_y, class_test_y])
    # print(total_test_y)
    # print(total_class_test_y)
    test_cm = eval.confusion_matrix(total_test_y, total_class_test_y, class_train_prob_y.columns)

    DataViz.plot_confusion_matrix(test_cm, class_train_prob_y.columns, normalize=False)




if __name__ == "__main__":
    main()