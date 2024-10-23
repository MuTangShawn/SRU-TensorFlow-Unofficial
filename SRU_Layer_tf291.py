
"""Simple Recurrent Unit layer."""

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine import base_layer
from keras.engine.input_spec import InputSpec
from keras.layers.rnn import rnn_utils
from keras.layers.rnn.base_rnn import RNN
from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.utils import tf_utils
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')

@keras_export(v1=['keras.layers.SRUCell'])
class SRUCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
  """Cell class for the SRU layer.
  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et al., 2015](
        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_activation=False,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='he_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               reset_after=True,
               **kwargs):
    if units < 0:
      raise ValueError(f'Received an invalid value for argument `units`, '
                       f'expected a positive integer, got {units}.')
    # By default use cached variable under v2 mode, see b/143699808.
    if tf.compat.v1.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(SRUCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias
    self.use_activation = use_activation

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    implementation = kwargs.pop('implementation', 1)
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    self.reset_after = reset_after
    self.state_size = [self.units,]
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]
    default_caching_device = rnn_utils.caching_device(self)

    self.equal_units = True if(input_dim == self.units) else False

    if self.equal_units:
      self.kernal_shape = (input_dim, self.units * 3)
    else:
      self.kernal_shape = (input_dim, self.units * 4)

    self.kernel = self.add_weight(
        shape=self.kernal_shape,
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)

    self.vector = self.add_weight(
        shape=(self.units, 2),
        name='vector',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if not self.reset_after:
        bias_shape = (2 * self.units,)
      else:
        bias_shape = (2, 2 * self.units)
      self.bias = self.add_weight(shape=bias_shape,
                                  name='bias',
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    c_tm1 = states[0]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(c_tm1, training, count=2) # not use

    if self.use_bias:
      if not self.reset_after:
        input_bias, recurrent_bias = self.bias, None
      else:
        input_bias, recurrent_bias = tf.unstack(self.bias)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_f = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_h = inputs
      else:
        inputs_f = inputs
        inputs_r = inputs
        inputs_c = inputs
        inputs_h = inputs

      if 0. < self.recurrent_dropout < 1.:
        c_tm1_f = c_tm1 * rec_dp_mask[0]
        c_tm1_r = c_tm1 * rec_dp_mask[1]
      else:
        c_tm1_f = c_tm1
        c_tm1_r = c_tm1

      if self.equal_units:
        x_f = backend.dot(inputs_f, self.kernel[:, :self.units])
        x_r = backend.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
        x_c = backend.dot(inputs_c, self.kernel[:, self.units * 2:])
        x_h = inputs_h
      else:
        x_f = backend.dot(inputs_f, self.kernel[:, :self.units])
        x_r = backend.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
        x_c = backend.dot(inputs_c, self.kernel[:, self.units * 2:self.units * 3])
        x_h = backend.dot(inputs_h, self.kernel[:, self.units * 3:])

      if self.use_bias:
        x_f = backend.bias_add(x_f, input_bias[:self.units])
        x_r = backend.bias_add(x_r, input_bias[self.units:])

      recurrent_f = tf.multiply(c_tm1_f, self.vector[:, 0])
      recurrent_r = tf.multiply(c_tm1_r, self.vector[:, 1])

      if self.reset_after and self.use_bias:
        recurrent_f = backend.bias_add(recurrent_f, recurrent_bias[:self.units])
        recurrent_r = backend.bias_add(recurrent_r, recurrent_bias[self.units:])
      
      f = self.recurrent_activation(x_f + recurrent_f)
      r = self.recurrent_activation(x_r + recurrent_r)

      c = f * c_tm1 + (1 - f) * x_c
      if self.use_activation:
        h = r * self.activation(c) + (1 - r) * x_h
      else:
        h = r * c + (1 - r) * x_h

    else:
      if 0. < self.dropout < 1.:
        inputs_drop = inputs * dp_mask[0]
        matrix_x = backend.dot(inputs_drop, self.kernel)
      else:
        matrix_x = backend.dot(inputs, self.kernel)

      if self.equal_units:
        x_f, x_r, x_c = tf.split(matrix_x, 3, axis=-1)
        x_h = inputs
      else:
        x_f, x_r, x_c, x_h = tf.split(matrix_x, 4, axis=-1)

      if self.use_bias:
        x_f = backend.bias_add(x_f, input_bias[:self.units])
        x_r = backend.bias_add(x_r, input_bias[self.units:])
      
      recurrent_f = tf.multiply(c_tm1, self.vector[:, 0])
      recurrent_r = tf.multiply(c_tm1, self.vector[:, 1])

      if self.reset_after and self.use_bias:
        recurrent_f = backend.bias_add(recurrent_f, recurrent_bias[:self.units])
        recurrent_r = backend.bias_add(recurrent_r, recurrent_bias[self.units:])

      f = self.recurrent_activation(x_f + recurrent_f)
      r = self.recurrent_activation(x_r + recurrent_r)

      c = f * c_tm1 + (1 - f) * x_c
      if self.use_activation:
        h = r * self.activation(c) + (1 - r) * x_h
      else:
        h = r * c + (1 - r) * x_h

      new_state = [c] if tf.nest.is_nested(states) else c
    return h, new_state

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'use_activation':
            self.use_activation,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after,
    }
    config.update(rnn_utils.config_for_enable_caching_device(self))
    base_config = super(SRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return rnn_utils.generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype)

@keras_export(v1=['keras.layers.SRU'])
class SRU(DropoutRNNCellMixin, RNN, base_layer.BaseRandomLayer):
  """
  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    reset_after: SRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before" (default),
      True = "after" (cuDNN compatible).
  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False`
      entry indicates that the corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               use_activation=False,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               time_major=False,
               reset_after=True,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.
    self._return_runtime = kwargs.pop('return_runtime', False)
    implementation = kwargs.pop('implementation', 2)
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=2`.'
                      'Please update your layer call.')
    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}
    cell = SRUCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        use_activation=use_activation,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        reset_after=reset_after,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)
    super(SRU, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        time_major=time_major,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]
   

  def call(self, inputs, mask=None, training=None, initial_state=None):
    return super(SRU, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def use_activation(self):
    return self.cell.use_activation

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  @property
  def reset_after(self):
    return self.cell.reset_after

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'use_activation':
            self.use_activation,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after
    }
    config.update(rnn_utils.config_for_enable_caching_device(self.cell))
    base_config = super(SRU, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)
