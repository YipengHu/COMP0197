import os
import json
import time
import tensorflow as tf

from data import load_text
from tokeniser import build_char_vocab, encode
from utils_tf import get_batch, CharTransformer

def main():
    # Files
    data_path = os.path.join("data", "tinyshakespeare.txt")
    weights_path = os.path.join("ckpt_tf_weights")   # TensorFlow will create multiple files with this prefix
    meta_path = os.path.join("ckpt_tf_meta.json")

    # Hyperparameters
    block_size = 128
    batch_size = 64
    n_layer = 4
    n_head = 4
    n_embd = 128
    dropout = 0.1
    lr = 3e-4

    max_steps = 1200
    eval_every = 200
    seed = 42

    tf.random.set_seed(seed)

    text = load_text(data_path)
    vocab = build_char_vocab(text)
    ids = tf.constant(encode(text, vocab.stoi), dtype=tf.int32)

    n = int(0.9 * int(ids.shape[0]))
    train_data = ids[:n]
    val_data = ids[n:]

    config = dict(
        vocab_size=vocab.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )

    model = CharTransformer(**config)
    opt = tf.keras.optimizers.AdamW(learning_rate=lr)

    # Build variables
    _ = model(tf.zeros([1, 1], dtype=tf.int32), training=False)

    @tf.function
    def loss_fn(x, y, training=True):
        logits = model(x, training=training)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(y, [-1]),
            logits=tf.reshape(logits, [-1, config["vocab_size"]]),
        )
        return tf.reduce_mean(loss)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(x, y, training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def estimate_loss():
        def mean_loss(d):
            losses = []
            for _ in range(20):
                xb, yb = get_batch(d, block_size, batch_size)
                losses.append(float(loss_fn(xb, yb, training=False).numpy()))
            return sum(losses) / len(losses)
        return mean_loss(train_data), mean_loss(val_data)

    t0 = time.time()
    for step in range(1, max_steps + 1):
        xb, yb = get_batch(train_data, block_size, batch_size)
        _ = train_step(xb, yb)

        if step == 1 or step % eval_every == 0:
            tr, va = estimate_loss()
            print(f"step {step:4d}, train {tr:.4f}, val {va:.4f}, elapsed {time.time()-t0:.1f}s")

    model.save_weights(weights_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"chars": vocab.chars, "config": config}, f)

    print(f"Saved weights prefix: {weights_path}")
    print(f"Saved meta:           {meta_path}")

if __name__ == "__main__":
    main()
