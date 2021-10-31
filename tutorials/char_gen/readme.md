# Character generation

This tutorial trains a recurrent neural network to generating names from a specified languages. This is adapted from [https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html).

This tutorial requires to download the [data](https://download.pytorch.org/tutorial/data.zip) and extract the downloaded file in the current directory.

It might be helpful to read first a simpler tutorial for name classification from the same author: [https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html). A copy of the classification tutorial is in the `char_cls_pt.py`

Example generated names:
```bash
>>> samples('Russian', 'RUSSIAN')
Rovantov
Uantov
Shavavav
Shanton
Iantonov
Ander
Nantonov
>>> samples('German', 'GERMAN')
Ganter
Eren
Romer
Martent
Allen
Nert
>>> samples('Spanish', 'SPANISH')
Serta
Paner
Allan
Naner
Iare
Santer
Haner
>>> samples('Chinese', 'CHINESE')
Chan
Hon
Iun
Nin
Eun
Shing
Eun
```