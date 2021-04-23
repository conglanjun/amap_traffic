import tensorflow as tf
from model.transformer.Encoder import Encoder
from model.transformer.Decoder import Decoder


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, rate=0.1):
    super(Transformer, self).__init__()

    self.tokenizer = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

    # self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inp, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

    enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    # dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    # final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output

if __name__ == '__main__':
    sample_transformer = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=2048, target_vocab_size=2048, pe_input=1280)

    temp_input = tf.random.uniform((5, 1280), dtype=tf.int64, minval=0, maxval=200)

    fn_out = sample_transformer(temp_input, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)