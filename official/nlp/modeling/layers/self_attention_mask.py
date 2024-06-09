# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keras layer that creates a self-attention mask."""
from typing import Optional
import tensorflow as tf, tf_keras


def get_mask(inputs: tf.Tensor,
             to_mask: tf.Tensor,
             dtype: Optional[tf.DType] = None) -> tf.Tensor:
  """Gets a 3D self-attention mask.

  Args:
    inputs: from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length,
      ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    dtype: the output Tensor dtype.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = tf.shape(inputs)
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]
  dtype = inputs.dtype if dtype is None else dtype

  to_shape = tf.shape(to_mask)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), dtype=dtype)

  # 广播遵循以下规则：
  #1. 如果输入张量的维度数小于目标形状的维度数，则在输入张量的前面添加维度，使其维度数与目标形状相同。
  #2. 如果输入张量在某个维度上的大小为 1，则可以在该维度上进行广播，即将该维度上的值复制到目标形状的大小。
  #3. 如果输入张量在某个维度上的大小不为 1 且与目标形状不匹配，则会引发错误。
  return tf.broadcast_to(to_mask, [batch_size, from_seq_length, to_seq_length])


@tf_keras.utils.register_keras_serializable(package='Text')
class SelfAttentionMask(tf_keras.layers.Layer):
  """Create 3D attention mask from a 2D tensor mask.

    inputs[0]: from_tensor: 2D or 3D Tensor of shape
      [batch_size, from_seq_length, ...].
    inputs[1]: to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """

  def call(self, inputs, to_mask=None):
    if isinstance(inputs, list) and to_mask is None:
      to_mask = inputs[1]
      inputs = inputs[0]
    return get_mask(inputs, to_mask)
