#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/11/19

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd

def load_data(path, row, col):
    return pd.read_csv(path, 
                       names = [row, col, 'Data'],
                       header = None,
                       skiprows=1, 
                       dtype={0:str, 1:str, 2:str})


def _loadICM_genres(genres_path, separator=','):

    ICM_genres_dataframe = load_data(genres_path, 'ItemID', 'FeatureID')
    
    return ICM_genres_dataframe

def _loadICM_subgenres(subgenre_path, separator=','):

    ICM_subgenre_dataframe = load_data(subgenre_path, 'ItemID', 'FeatureID')
    
    return ICM_subgenre_dataframe

def _loadICM_channel(channel_path, separator=','):

    ICM_channel_dataframe = load_data(channel_path, 'ItemID', 'FeatureID')
    
    return ICM_channel_dataframe

def _loadICM_event(event_path, separator=','):

    ICM_event_dataframe = load_data(event_path, 'ItemID', 'FeatureID')
    
    return ICM_event_dataframe


def _loadURM(URM_path,  separator=','):

    URM_all_dataframe = load_data(URM_path, 'UserID', 'ItemID')

    return URM_all_dataframe






