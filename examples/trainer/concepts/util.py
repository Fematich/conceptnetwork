#!/usr/bin/python
"""Reusable utility functions.

This file is generic and can be reused by other models without modification.
"""

import tensorflow as tf


class DefaultToKeyDict(dict):
  """Custom dictionary to use the key as the value for any missing entries."""

  def __missing__(self, key):
    return str(key)


def int64_feature(value):
  """Create a multi-valued int64 feature from a single value."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  """Create a multi-valued bytes feature from a single value."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
  """Create a multi-valued float feature from a single value."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))