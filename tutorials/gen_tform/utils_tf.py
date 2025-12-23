import tensorflow as tf

def get_batch(data: tf.Tensor, block_size: int, batch_size: int):
    max_start = tf.shape(data)[0] - block_size - 1
    ix = tf.random.uniform([batch_size], minval=0, maxval=max_start, dtype=tf.int32)

    xs, ys = [], []
    for i in tf.unstack(ix):
        xs.append(data[i : i + block_size])
        ys.append(data[i + 1 : i + block_size + 1])

    return tf.stack(xs, axis=0), tf.stack(ys, axis=0)

class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = tf.keras.layers.Dense(3 * n_embd, use_bias=False)
        self.proj = tf.keras.layers.Dense(n_embd, use_bias=False)
        self.drop = tf.keras.layers.Dropout(dropout)

    def _split_heads(self, x):
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.n_head, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        return tf.reshape(x, [b, t, self.n_embd])

    def call(self, x, training=False):
        t = tf.shape(x)[1]
        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, axis=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scale = tf.cast(self.head_dim, tf.float32) ** -0.5
        att = tf.matmul(q, k, transpose_b=True) * scale  # (B, nh, T, T)

        mask = tf.linalg.band_part(tf.ones([t, t], dtype=tf.float32), -1, 0)
        att = att + (1.0 - mask)[tf.newaxis, tf.newaxis, :, :] * tf.constant(-1e9, tf.float32)

        att = tf.nn.softmax(att, axis=-1)
        att = self.drop(att, training=training)

        out = tf.matmul(att, v)
        out = self._merge_heads(out)
        out = self.proj(out)
        out = self.drop(out, training=training)
        return out

class MLP(tf.keras.layers.Layer):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(4 * n_embd)
        self.fc2 = tf.keras.layers.Dense(n_embd)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = tf.nn.gelu(x)
        x = self.fc2(x)
        return self.drop(x, training=training)

class Block(tf.keras.layers.Layer):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = MLP(n_embd, dropout)

    def call(self, x, training=False):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x

class CharTransformer(tf.keras.Model):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_emb = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.pos_emb = tf.keras.layers.Embedding(block_size, n_embd)
        self.drop = tf.keras.layers.Dropout(dropout)

        self.blocks = [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = tf.keras.layers.Dense(vocab_size, use_bias=False)

    def call(self, idx, training=False):
        t = tf.shape(idx)[1]
        pos = tf.range(0, t, dtype=tf.int32)[tf.newaxis, :]
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x, training=training)
        for blk in self.blocks:
            x = blk(x, training=training)
        x = self.ln_f(x)
        return self.head(x)

    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        idx = tf.convert_to_tensor(idx, dtype=tf.int32)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond, training=False)
            logits = logits[:, -1, :] / temperature
            probs = tf.nn.softmax(logits, axis=-1)
            next_token = tf.random.categorical(tf.math.log(probs), num_samples=1)
            idx = tf.concat([idx, tf.cast(next_token, tf.int32)], axis=1)
        return idx
