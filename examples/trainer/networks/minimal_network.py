#!/usr/bin/python

import logging
import tensorflow as tf

from conceptnetwork import Network, Concept
from concepts.minimal_concept import MinimalConcept


class MinimalNetwork(Network):
    """MinimalNetwork is the Network class illustrating how to work with
    ConceptNetwork Networks"""
    version = "1.0"

    def __init__(self):
        super(MinimalNetwork, self).__init__()
        self.concepts["origin"] = MinimalConcept()
        self.concepts["target"] = MinimalConcept(target=True)

    def _get_test_input(self):
        str1 = (open('../data/example.xml', 'r').readlines()
                [0]).decode('utf-8')
        return (str1, str1)

    def preprocess(self, raw_input):
        """This function is responsible for encoding a raw_input object to a
        TensorFlow Example object"""
        features = dict()
        for i, cname in enumerate(["origin", "target"]):
            concept = self.concepts[cname]
            cfeatures = concept.preprocess(raw_input[i].encode('utf-8'))
            for k, v in cfeatures.items():
                features[cname + '_' + k] = v
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        return example

    def inference(self, features):
        features_len = int(features["origin"].get_shape()[1])
        embedding = tf.layers.dense(
            inputs=features["origin"], units=100, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=embedding, units=features_len)
        return {"logits": logits}

    def loss(self, predictions, labels):
        loss = tf.losses.mean_squared_error(
            predictions["logits"], labels["target"])
        return loss
