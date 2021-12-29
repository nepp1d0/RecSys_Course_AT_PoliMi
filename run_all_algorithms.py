
from Recommenders.Recommender_import_list import *
import scipy.sparse as sps

from Data_manager.ChallengeDataset.ChallengeDataset import ChallengeDataset
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from Evaluation.Evaluator import EvaluatorHoldout
import traceback, os


def _get_instance(recommender_class, URM_train, ICM_all):

    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object

if __name__ == '__main__':


    dataset_object = ChallengeDataset()

    dataSplitter = DataSplitter_leave_k_out(dataset_object, k_out_value=2,force_new_split = True)

    dataSplitter.load_data()
    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    ICM_genres = dataSplitter.get_loaded_ICM_dict()["ICM_genre"]
    ICM_subgenres = dataSplitter.get_loaded_ICM_dict()["ICM_subgenre"]
    ICM_channel = dataSplitter.get_loaded_ICM_dict()["ICM_channel"]
    ICM_event = dataSplitter.get_loaded_ICM_dict()["ICM_event"]




    #Stack URM and ICMs
    stacked_URM = sps.vstack([URM_train, ICM_genres.T])
    stacked_URM = sps.vstack([stacked_URM, ICM_subgenres.T])
    stacked_URM = sps.vstack([stacked_URM, ICM_channel.T])
    stacked_URM = sps.vstack([stacked_URM, ICM_event.T])
    stacked_URM = sps.csr_matrix(stacked_URM)

    recommender_class_list = [  
        MultiThreadSLIM_SLIMElasticNetRecommender
        ]


    evaluator = EvaluatorHoldout(URM_test, [5, 20], exclude_seen=True)

    # from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch

    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": EvaluatorHoldout(URM_validation, [5, 10, 20], exclude_seen=True),
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP",
                              }


    output_root_path = "./result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    logFile = open(output_root_path + "result_all_algorithms.txt", "a")


    for recommender_class in recommender_class_list:

        try:

            print("Algorithm: {}".format(recommender_class))

            recommender_object = _get_instance(recommender_class, stacked_URM, ICM_genres)

            optimal_params= { 'alpha' : 0.007596772625600448,
                                'l1_ratio': 0.09354638545594608,
                                'topK': 410,
                                'workers': 6 }

            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {  **optimal_params,
                                **earlystopping_keywargs}
            else:
                fit_params = {**optimal_params}

            recommender_object.fit(**fit_params)

            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)

            recommender_object.save_model(output_root_path, file_name = "temp_model.zip")

            recommender_object = _get_instance(recommender_class, URM_train, ICM_genres)
            recommender_object.load_model(output_root_path, file_name = "temp_model.zip")

            os.remove(output_root_path + "temp_model.zip")

            '''  results_run_2, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

            if recommender_class not in [Random]:
                assert results_run_1.equals(results_run_2)'''

            print("Algorithm: {}, results: \n{}".format(recommender_class, results_run_string_1))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender_class, results_run_string_1))
            logFile.flush()


        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()
