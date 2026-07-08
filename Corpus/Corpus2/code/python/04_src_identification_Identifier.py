import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy.spatial import distance
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition  # PCA
from sklearn.metrics import confusion_matrix

import json

import ml.Features as ft
from utils import Utils

class Identifier(object):

    def __init__(self):
        columns = ['mean_height', 'min_height', 'max_height', 'mean_width', 'min_width', 'max_width', 'time', 'girth','id']
        self.data = DataFrame(columns=columns)
        self.event = []
    @staticmethod
    def subscribe(ch, method, properties, body):
        """
        prints the body message. It's the default callback method
        :param ch: keep null
        :param method: keep null
        :param properties: keep null
        :param body: the message
        :return:
        """
        #first we get the JSON from body

        #we check if it's part of the walking event

        #if walking event is completed, we


if __name__ == '__main__':
    # we setup needed params
    MAX_HEIGHT = 203
    MAX_WIDTH = 142
    SPEED = 3
    SAMPLING_RATE = 8
    mq_host = '172.26.56.122'
    queue_name = 'door_data'
    # setting up MQTT subscriber
    Utils.sub(queue_name=queue_name,callback=subscribe,host=mq_host)