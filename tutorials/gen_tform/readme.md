# Transformer Tutorials for Generative Models 

This folder contains two self-contained, minimal implementations of a decoder-only Transformer (GPT-style) with causal self-attention for character-level language modelling.

Both versions:
- download a tiny text dataset automatically (**Tiny Shakespeare**),
- build a simple character vocabulary,
- train a small Transformer,
- save a checkpoint,
- load the trained model and generate text.

## Requirements
```bash
micromamba activate comp0197 
```

## Files

### Shared (used by both)
- `data.py`  
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

An example of generated text (they do not make sense with this a small-scale demo, but feel Shakespearean):

```bash
>>> python test_tf.py
ROMEO:
With tingued in of thy fold ang's my jove.
Ray, Cut How, whout:
Be sine mack the nad so be hintith.

Servize:
I livestiond love the mere the comers, moak,
I hatte laygend, and wo mingook.

WANGORD II:
I a that is whys as hery it thou speell, beit,
Your liied of a ming the firent.

CARIUS:
And agaimiiend, not, be the tuze, in is thee brove,
Os this that son to nest,
Isep with wich that the beingst for nowst:
In you thou the is cwad now, her sovive madter,
For me andlare to faitin; for, it your he
```
