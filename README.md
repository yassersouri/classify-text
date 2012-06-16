<center>Salam</center>

## Text Classification with python

This is an experiment. We want to classify text with python.

### Dataset

For dataset I used the famous "Twenty Newsgrousps" dataset. You can find the dataset freely [here](http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups). 

I've included a subset of the dataset in the repo, located at `dataset\` directory. This subset includes 6 of the 20 newsgroups: `space`, `electronics`, `crypt`, `hockey`, `motorcycles` and `forsale`.

We you run `main.py` it asks you for the root of the dataset. You can supply your own dataset assuming it has a similar directry structure.

#### UTF-8 incompatibility

Some of the supplied text files had incompatibility with utf-8!

Even textedit.app can't open those files. And they created problem in the code. So I'll delete them as part of the preprocessing.

### Requirements

* python 2.7

* python modules:

  - scikit-learn (v 0.11)
  - scipy (v 0.10.1)
  - colorama
  - termcolor

### The code

The code is pretty straight forward and well documented.

#### Running the code

	python main.py

### Experiments

