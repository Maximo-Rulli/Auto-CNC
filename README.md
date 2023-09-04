# Auto-CNC
The objective of this project is to make more efficient an industry supply chain by implementing transformers in the sorting algorithm of which products should be produced and which ones should be delayed. The dataset consists of 1000 real life examples of sequences of product queries and as the output label we used the real product that was produced within the context (more detailed can be found inside the data_analysy.ipynb). We used an encoder model (the well-known BERT), to predict the product that should be produced based on a sequence of input products that were sorted by query date.

The project consists of the following files:
