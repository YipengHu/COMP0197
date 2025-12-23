# Minimal Transformer + Attention Tutorials (PyTorch + TensorFlow)

This folder contains two **self-contained**, **minimal** implementations of a **decoder-only Transformer** (GPT-style) with **causal self-attention** for **character-level language modelling**.

Both versions:
- download a tiny text dataset automatically (**Tiny Shakespeare**),
- build a simple character vocabulary,
- train a small Transformer,
- save a checkpoint,
- load the trained model and generate text.

No extra libraries beyond **PyTorch** or **TensorFlow** (plus Python standard library).

---

## Files

### Shared (used by both)
- `dataset.py`  
  Downloads and loads Tiny Shakespeare into a local `data/` folder.
- `tokeniser.py`  
  Minimal character vocabulary, `encode`, `decode`.

### PyTorch
- `utils_pt.py`  
  Batching (`get_batch`) + Transformer model definition.
- `train_pt.py`  
  Trains the model and saves `ckpt_pt.pt`.
- `test_pt.py`  
  Loads `ckpt_pt.pt` and generates text from a prompt.

### TensorFlow
- `utils_tf.py`  
  Batching (`get_batch`) + Transformer model definition.
- `train_tf.py`  
  Trains the model and saves TensorFlow weights + `ckpt_tf_meta.json`.
- `test_tf.py`  
  Loads weights + meta and generates text from a prompt.

---

## Requirements

- Python 3.9+ recommended
- Install frameworks (CPU is fine):

```bash
pip install torch tensorflow
