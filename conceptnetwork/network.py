#!/usr/bin/python
import logging
from collections import defaultdict
import abc


class Network(object):
    """Base class for Network"""

    def __init__(self):
        __metaclass__ = abc.ABCMeta
        self.concepts = dict()

    def _select_features(self, concept_key, features):
        """Select and translate features for this concept_key"""
        cfeatures = dict()
        for k, v in features.items():
            if k.startswith(concept_key):
                cfeatures['_'.join(k.split('_')[1:])] = v
        return cfeatures

    def get_vectors(self, features):
        vectors = dict()
        for cname, concept in self.concepts.items():
            vectors[cname] = concept.model(
                self._select_features(cname, features))
        return vectors

    def __repr__(self):
        return self.__class__.__name__ + self.version + '_' + '|'.join([str(c) for c in self.concepts.values()])

    @abc.abstractmethod
    def _get_test_input(self):
        """This function will return a test input for encoding,
        this function is only required for testing"""
        raise NotImplementedError(
            'The base class needs to implement "get_test_input"')

    @abc.abstractmethod
    def encode(self, raw_input):
        """This function is responsible for encoding an raw_input object to a 
        dict of feature names and a corresponding TensorFlow Example protobuffers

        Defaults to a concatenation of the encodings of the multiple concepts"""
        features = dict()
        for cname, concept in self.concepts.items():
            cfeatures = concept.encode(raw_input)
            for k, v in cfeatures.items():
                features[cname + '_' + k] = v
        return features

    @abc.abstractmethod
    def featdef(self):
        """This function defines thet interface between encode and model:
        Returns a dictionary of feature names and tf FeatureEncodingTypes
        Defaults to a concatenation of the encodings of the multiple concepts"""
        features_def = dict()
        for cname, concept in self.concepts.items():
            cfeatures_def = concept.featdef()
            for k, v in cfeatures_def.items():
                features_def[cname + '_' + k] = v
        return features_def

    @abc.abstractmethod
    def model(self, features):
        """This function takes a dictionary of tensors(features) and 
        specifies the Tensorflow operations to transform these into a single tensor"""
        raise NotImplementedError(
            'The base class needs to implement "model"')

    @classmethod
    def _test(cls):
        """test is resposible for testing a Network class"""
        import tensorflow as tf
        self = cls()
        logging.info('\n' + '*' * 50 + '\n' + '*' * 50)
        logging.info('\n' + '*' * 50 + '\n' + '*' * 50)
        logging.info('Test Network : %s', self)
        features = self.encode(self._get_test_input())
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        serialized_example = example.SerializeToString()
        logging.info('Successfully serialized tfrecord')

        reconstructed_features = tf.parse_single_example(
            serialized_example,
            features=self.featdef(),
            name='reconstruct_features')
        logging.info('Successfully reconstructed tfrecord')

        embedding = self.model(reconstructed_features)
        with tf.Session():
            tf.global_variables_initializer().run()
            vector = embedding.eval()
            logging.info('vector : %s', str(vector))
        logging.info('\n' + '*' * 50 + '\n' + '*' * 50)
