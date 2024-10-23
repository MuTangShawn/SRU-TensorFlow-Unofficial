import numpy as np
import tensorflow as tf
import SRU_Layer_tf2101

inputs = np.random.random((32, 10, 8))
sru = SRU_Layer_tf2101.SRU(4)
output = sru(inputs)
print(output.shape)
sru = SRU_Layer_tf2101.SRU(4, return_sequences=True, return_state=True)
whole_sequence_output, final_state = sru(inputs)
print(whole_sequence_output.shape)
print(final_state.shape)
bisru = tf.keras.layers.Bidirectional(SRU_Layer_tf2101.SRU(4, return_sequences=True), merge_mode="concat")
output = bisru(inputs)
print(output.shape)