#!/usr/bin/python
import logging
import abc
from collections import defaultdict
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


class Network(object):
    """Base class for Network"""

    def __init__(self):
        __metaclass__ = abc.ABCMeta
        self.concepts = dict()
        self._feature_names = set()
        self._target_names = set()

    @property
    def feature_names(self):
        if len(self._feature_names) < 1:
            self._targetandfeaturenames()
        return self._feature_names

    @property
    def target_names(self):
        if len(self._target_names) < 1:
            self._targetandfeaturenames()
        return self._target_names

    def _targetandfeaturenames(self):
        for cname, concept in self.concepts.items():
            if concept.target:
                self._target_names.add(cname)
            else:
                self._feature_names.add(cname)

    def _select_features(self, concept_key, features):
        """Select and translate features for this concept_key"""
        cfeatures = dict()
        for k, v in features.items():
            if k.startswith(concept_key):
                cfeatures['_'.join(k.split('_')[1:])] = v
        return cfeatures

    def get_featurevectors(self, features):
        vectors = dict()
        for cname, concept in self.concepts.items():
            if not concept.target:
                vectors[cname] = concept.inference(
                    self._select_features(cname, features))
        return vectors

    def get_targetvectors(self, features):
        vectors = dict()
        for cname, concept in self.concepts.items():
            if concept.target:
                vectors[cname] = concept.inference(
                    self._select_features(cname, features))
        return vectors

    def __repr__(self):
        return self.__class__.__name__ + self.version + '__' + \
            '_'.join([str(c) for c in self.concepts.values()])

    @abc.abstractmethod
    def _get_test_input(self):
        """This function will return a test input for encoding,
        this function is only required for testing"""
        raise NotImplementedError(
            'The base class needs to implement "get_test_input"')

    @abc.abstractmethod
    def preprocess(self, raw_input):
        """This function is responsible for encoding an raw_input object to a
        TensorFlow Example object
        Defaults to a concatenation of the encodings of the multiple
        concepts"""
        features = dict()
        for cname, concept in self.concepts.items():
            cfeatures = concept.preprocess(raw_input)
            for k, v in cfeatures.items():
                features[cname + '_' + k] = v
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        return example

    @abc.abstractmethod
    def featdef(self):
        """This function defines thet interface between preprocess and inference:
        Returns a dictionary of feature names and tf FeatureEncodingTypes
        Defaults to a concatenation of the encodings of the multiple
        concepts"""
        features_def = dict()
        for cname, concept in self.concepts.items():
            cfeatures_def = concept.featdef()
            for k, v in cfeatures_def.items():
                features_def[cname + '_' + k] = v
        return features_def

    @abc.abstractmethod
    def inference(self, features):
        """This function takes a dictionary of tensors(features) and
        returns the logits calculate with this network"""
        raise NotImplementedError(
            'The base class needs to implement "inference"')

    @abc.abstractmethod
    def loss(self, predictions, labels):
        """This function calculates the loss between the labels and the predictions
        (output from inference)"""
        raise NotImplementedError(
            'The base class needs to implement "loss"')

    def feature_engineering_fn(self, features, labels):
        return self.get_featurevectors(features),\
            self.get_targetvectors(labels)

    def train(self, loss):
        """This function takes a dictionary of tensors(features) and
        specifies the Tensorflow operations to transform these into a single tensor"""
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.1,
            optimizer='Adagrad')
        return train_op

    def build_model_fn(self):
        """Build model function.

        Returns:
          A model function that can be passed to `Estimator` constructor.
        """
        def _model_fn(features, labels, mode):
            """Creates the prediction and its loss.

            Args:
              features: A dictionary of tensors keyed by the feature name.
              labels: A tensor representing the labels.
              mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.

            Returns:
              A tuple consisting of the prediction, loss, and train_op.
            """
            predictions = self.inference(features)
            if mode == tf.contrib.learn.ModeKeys.INFER:
                return predictions, None, None

            loss = self.loss(predictions, labels)
            if mode == tf.contrib.learn.ModeKeys.EVAL:
                return predictions, loss, None

            train_op = self.train(loss)
            return predictions, loss, train_op

        return _model_fn

    def build_input_fn(self, batch_size, mode, input_dir):
        """Build input function.

        Args:
          batch_size: Batch size
          mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.
          input_dir: path containing the input files

        Returns:
          Tuple, dictionary of feature column name to tensor and labels.
        """
        def _input_fn():
            """Supplies the input to the model.

            Returns:
              A tuple consisting of 1) a dictionary of tensors whose keys are
              the feature names, and 2) a tensor of target labels if the mode
              is not INFER (and None, otherwise).
            """
            input_files = sorted(list(tf.gfile.Glob(input_dir)))
            logging.info("Reading files from %s", input_dir)
            include_target_column = (mode != tf.contrib.learn.ModeKeys.INFER)

            def gzip_reader():
                return tf.TFRecordReader(
                    options=tf.python_io.TFRecordOptions(
                        compression_type=TFRecordCompressionType.GZIP))
            reader_fn = gzip_reader

            all_features = tf.contrib.learn.io.read_batch_features(
                file_pattern=input_dir,
                batch_size=batch_size,
                queue_capacity=3 * batch_size,
                randomize_input=mode == tf.contrib.learn.ModeKeys.TRAIN,
                feature_queue_capacity=5,
                reader=reader_fn,
                features=self.featdef())
            target = dict()
            features = dict()
            for f_name, feature in all_features.iteritems():
                c_name = f_name.split('_')[0]
                if c_name in self.target_names:
                    target[f_name] = feature
                else:
                    features[f_name] = feature
            # print('features')
            # print(features)
            # print('target')
            # print(target)
            if len(target) < 1:
                target = None
            return features, target

        return _input_fn

    @classmethod
    def _test(cls):
        """test is resposible for testing a Network class"""
        self = cls()
        logging.info('\n' + '*' * 50 + '\n' + '*' * 50)
        logging.info('\n' + '*' * 50 + '\n' + '*' * 50)
        logging.info('Test Network : %s', self)
        example = self.preprocess(self._get_test_input())
        serialized_example = example.SerializeToString()
        logging.info('Successfully serialized tfrecord')

        reconstructed_features = tf.parse_single_example(
            serialized_example,
            features=self.featdef(),
            name='reconstruct_features')
        logging.info('Successfully reconstructed tfrecord')

        target = dict()
        features = dict()

        for f_name, feature in reconstructed_features.iteritems():
            c_name = f_name.split('_')[0]
            if c_name in self.target_names:
                target[f_name] = feature
            else:
                features[f_name] = feature
        features, labels = self.feature_engineering_fn(features, target)
        new_features = dict()
        for k, v in features.iteritems():
            v_length = int(v.get_shape()[0])
            new_features[k] = tf.reshape(v, (v_length, 1))
        new_labels = dict()
        for k, v in labels.iteritems():
            v_length = int(v.get_shape()[0])
            new_labels[k] = tf.reshape(v, (v_length, 1))
        predictions = self.inference(new_features)
        loss = self.loss(predictions, new_labels)
        with tf.Session():
            tf.global_variables_initializer().run()
            vector = predictions['logits'].eval()
            logging.info('vector : %s', str(vector))
            losses = loss.eval()
            logging.info('loss : %s', str(losses))
