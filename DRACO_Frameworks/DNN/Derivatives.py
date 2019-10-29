import os
import sys
import numpy as np
import json

# local imports
filedir  = os.path.dirname(os.path.realpath(__file__))
DRACOdir = os.path.dirname(filedir)
basedir  = os.path.dirname(DRACOdir)
sys.path.append(basedir)

# imports with keras
import data_frame
import pandas as pd
# Limit gpu usage
import tensorflow as tf

class Inputs(object):
    def __init__(self, names, datatype=tf.float32, variable_scope="inputs"):
        self._names = names
        self._datatype = datatype
        self._variable_scope = variable_scope

        self._placeholders_list = []
        self._placeholders_dict = {}
        with tf.variable_scope(variable_scope):
            for name in names:
                if "[" in name:
                    name = name.replace("[","_")
                if "]" in name:
                    name = name.replace("]","")
                print name
               
                placeholder = tf.placeholder(
                    dtype=self._datatype, shape=(None, ), name=name)
                self._placeholders_list.append(placeholder)
                self.placeholders_dict[name] = placeholder

            self._placeholders = tf.stack(self._placeholders_list, axis=1)

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, x):
        raise Exception('Variable can not be set.')

    @property
    def datatype(self):
        return self._datatype

    @datatype.setter
    def datatype(self, x):
        raise Exception('Variable can not be set.')

    @property
    def placeholders(self):
        return self._placeholders

    @placeholders.setter
    def placeholders(self, x):
        raise Exception('Variable can not be set.')

    @property
    def placeholders_dict(self):
        return self._placeholders_dict

    @placeholders_dict.setter
    def placeholders_dict(self, x):
        raise Exception('Variable can not be set.')

    @property
    def variable_scope(self):
        return self._variable_scope

    @variable_scope.setter
    def variable_scope(self, x):
        raise Exception('Variable can not be set.')

class Outputs(object):
    def __init__(self, function, names, variable_scope="outputs"):
        self._names = names
        self._variable_scope = variable_scope

        if not function.shape[1] == len(names):
            raise Exception(
                'Shape of function does not match number of given names.')

        with tf.variable_scope(variable_scope):
            self._outputs_dict = {}
            for name, tensor in zip(names,
                                    tf.split(function, len(names), axis=1)):

                self._outputs_dict[name] = tf.identity(tensor, name=name)

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, x):
        raise Exception('Variable can not be set.')

    @property
    def variable_scope(self):
        return self._variable_scope

    @variable_scope.setter
    def variable_scope(self, x):
        raise Exception('Variable can not be set.')

    @property
    def outputs_dict(self):
        return self._outputs_dict

    @outputs_dict.setter
    def outputs_dict(self, x):
        raise Exception('Variable can not be set.')

class Derivatives(object):
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    def get(self, output, inputs):
        if not output in self._outputs.names:
            raise Exception('Output {} is not in list {}.'.format(
                output, self._outputs.names))
        for name in inputs:
            if not name in self._inputs.names:
                raise Exception('Input {} is not in list {}.'.format(
                    name, self._inputs.names))

        derivative = self._outputs.outputs_dict[output]
        for name in inputs:
            if "[" in name:
                name = name.replace("[","_")
            if "]" in name:
                name = name.replace("]","")
            derivative = tf.gradients(derivative,
                                      self._inputs.placeholders_dict[name])

        return derivative
