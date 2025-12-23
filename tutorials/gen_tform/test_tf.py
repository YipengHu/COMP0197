import os
import json
import tensorflow as tf

from tokeniser import decode
from utils_tf import CharTransformer

def main():
    weights_path = os.path.join("ckpt_tf_weights")
    meta_path = os.path.join("ckpt_tf_meta.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chars = meta["chars"]
    config = meta["config"]
    stoi = {ch: i for i, ch in enumerate(chars)}

    model = CharTransformer(**config)
    _ = model(tf.zeros([1, 1], dtype=tf.int32), training=False)
    model.load_weights(weights_path)

    prompt = "ROMEO:\n"
    idx0 = tf.constant([[stoi[c] for c in prompt]], dtype=tf.int32)

    out = model.generate(idx0, max_new_tokens=500, temperature=0.9)[0].numpy().tolist()
    print(decode(out, chars))

if __name__ == "__main__":
    main()
