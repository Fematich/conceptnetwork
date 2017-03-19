#!/usr/bin/python

"""Module that contains all candidate related concepts"""
import logging
from collections import defaultdict
import tensorflow as tf


from conceptnetwork import Concept
from concepts import util

class MinimalConcept(Concept):
    """MinimalConcept is the Concept class illustrating how to work with
    ConceptNetwork Concepts"""
    version = "1.0"

    # Normalize over possible sex/gender values.
    GENDER_MAP = defaultdict(lambda: Concept.NA_INTEGER)
    GENDER_MAP.update({
        'male': 0,
        'female': 1,
        'Male': 0,
        'Female': 1,
        'm': 0,
        'f': 1,
        'M': 0,
        'F': 1
    })

    def _get_test_input(self):
        return "123,m"

    def preprocess(self, raw_input):
        str_id, str_sex = raw_input.split(",")
        id = int(str_id)
        sex = self.GENDER_MAP[str_sex]
        features = {
            'candidate_id':
                util.float_feature(id),
            'gender':
                util.float_feature(sex),
        }
        return features

    def featdef(self):
        return {
            'candidate_id': tf.FixedLenFeature([], tf.float32),
            'gender': tf.FixedLenFeature([], tf.float32),
        }

    def inference(self, features):
        candidate_id = tf.cast(features['candidate_id'], tf.float32)
        gender = tf.cast(features['gender'], tf.float32)
        vector = tf.stack([candidate_id, gender], axis=0)
        return vector
