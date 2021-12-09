#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd
import zipfile, shutil, os
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.ChallengeDataset._utils_Challenge_parser import _loadURM, _loadICM_genres, _loadICM_subgenres, _loadICM_channel, _loadICM_event


class ChallengeDataset(DataReader):

    #DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DATASET_SUBFOLDER = "ChallengeDataset/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_genre", "ICM_channel", "ICM_subgenre", "ICM_event"]
    #AVAILABLE_UCM = ["UCM_all"]
    REPO_NAME = 'recommender-system-2021-challenge-polimi'

    IS_IMPLICIT = True

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original
        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "recommender-system-2021-challenge-polimi.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file.")

            #download_from_URL(self.DATASET_URL, zipFile_path, "ml-1m.zip")

            #dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")

        dataFile.extractall(path = os.path.join(zipFile_path, self.REPO_NAME))

        ICM_genre_path = os.path.join(zipFile_path ,self.REPO_NAME , 'data_ICM_genre.csv')
        ICM_channel_path = os.path.join(zipFile_path , self.REPO_NAME, 'data_ICM_channel.csv')
        ICM_event_path = os.path.join(zipFile_path , self.REPO_NAME, 'data_ICM_event.csv')
        ICM_subgenres_path = os.path.join(zipFile_path , self.REPO_NAME, 'data_ICM_subgenre.csv')

        #UCM_path = dataFile.extract("ml-1m/users.dat", path=zipFile_path + "decompressed/")
        URM_path = os.path.join(zipFile_path , self.REPO_NAME, 'data_train.csv')


        self._print("Loading Interactions")
        URM_all_dataframe = _loadURM(URM_path)

        self._print("Loading Item Features subgenres")
        ICM_subgenres_dataframe= _loadICM_subgenres(ICM_subgenres_path)

        self._print("Loading Item Features channel")
        ICM_channel_dataframe= _loadICM_channel(ICM_channel_path)

        self._print("Loading Item Features genres")
        ICM_genres_dataframe= _loadICM_genres(ICM_genre_path)

        self._print("Loading Item Features event")
        ICM_event_dataframe= _loadICM_event(ICM_event_path)


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_genres_dataframe, "ICM_genre")
        dataset_manager.add_ICM(ICM_subgenres_dataframe, "ICM_subgenres")
        dataset_manager.add_ICM(ICM_channel_dataframe, "ICM_channel")
        dataset_manager.add_ICM(ICM_event_dataframe, "ICM_event")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset

