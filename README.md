# Auto-CNC
The objective of this project is to improve a factory supply chain by implementing transformers in the sorting algorithm of which products should be produced and which ones should be delayed. The dataset consists of 1000 real-life examples of sequences of product queries and as the output label we used the real product that was produced within the context (more details can be found inside the data_analysy.ipynb). We used an encoder model (the well-known BERT), to predict the product that should be produced based on a sequence of input products that were sorted by query date.


## Project structure
The project consists of the following files:

* fine_tune.ipynb: file in which the BERT model is fine-tuned. With Google Colab and a T4 GPU, the model can be trained in under 15 minutes.

* data.h5: file in which training and testing data is stored, ready to be fed to the model.

* data_analysis.ipynb: file in which the preprocessing and dataset creation are mostly done (here is where the main logic and paradigm are stated).

* load_model.ipynb: file in which the fine-tuned model is loaded and its accuracy is assessed in depth.

* gaby_bert.py: test file in which we experimented with transformers.

* datasets: folder containing the raw data in csv format.


## Results
The model attains a good accuracy score, achieving a whopping 65% test accuracy considering the greatest logit, and an acceptable 85% accuracy on the test set when the top 3 logits of each prediction are considered. These results show that traditional RL algorithms may be well suited to scenarios like sorting in an infinite environment, but that transformers may also be useful and easy to implement compared to the logic and error analysis challenges that the design of an RL environment may have. Furthermore, transformers compared to RL paradigms may also be better suited to scenarios in which the reward function is difficult to design due to the lack of data, such as our case study, in which we lacked the data of an accurate measure to maximize or minimize (since no cost nor revenue data was present in the dataset).
