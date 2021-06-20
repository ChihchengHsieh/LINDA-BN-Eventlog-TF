import tensorflow as tf
from Transformer.attention import MultiHeadAttention
from Transformer.feed_forward import point_wise_feed_forward_network
from Transformer.positional_encode import positional_encoding

class PredictingNextTransformerEncoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
                pe_input, rate=0.1):
        super(PredictingNextTransformerEncoder, self).__init__()

        self.tokenizer = PredictNextEncoder(num_layers, d_model, num_heads, dff,
                                 vocab_size, pe_input, rate)


        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inp, training, combine_mask):

        # (batch_size, inp_seq_len, d_model)
        enc_output, attention_weights = self.tokenizer(inp, training, combine_mask)

        # (batch_size, inp_seq_len, target_vocab_size)
        final_output = self.final_layer(enc_output)

        return final_output, attention_weights

class PredictNextEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(PredictNextEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [PredictNextEncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        attention_weights = []
        for i in range(self.num_layers):
            x, attention_weight = self.enc_layers[i](x, training, mask)
            attention_weights.append(attention_weight)

        return x, attention_weights  # (batch_size, input_seq_len, d_model)


class PredictNextEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(PredictNextEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # (batch_size, input_seq_len, d_model)
        attn_output, attention_weight = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attention_weight
