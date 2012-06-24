<center>Salam</center>

## Text Classification with python

This is an experiment. We want to classify text with python.

### Dataset

For dataset I used the famous "Twenty Newsgrousps" dataset. You can find the dataset freely [here](http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups). 

I've included a subset of the dataset in the repo, located at `dataset\` directory. This subset includes 6 of the 20 newsgroups: `space`, `electronics`, `crypt`, `hockey`, `motorcycles` and `forsale`.

When you run `main.py` it asks you for the root of the dataset. You can supply your own dataset assuming it has a similar directory structure.

#### UTF-8 incompatibility

Some of the supplied text files had incompatibility with utf-8!

Even textedit.app can't open those files. And they created problem in the code. So I'll delete them as part of the preprocessing.

### Requirements

* python 2.7

* python modules:

  * scikit-learn (v 0.11)
  * scipy (v 0.10.1)
  * colorama
  * termcolor
  * matplotlib (for use in `plot.py`)

### The code

The code is pretty straight forward and well documented.

#### Running the code

	python main.py

### Experiments

For experiments I used the subset of the dataset (as described above). I assume that we like `hockey`, `crypt` and `electronics` newsgroups, and we dislike the others.

For each experiment we use a "feature vector", a "classifier" and a train-test splitting strategy.

#### Experiment 1: BOW - NB - 20% test

In this experiment we use a Bag Of Words (**BOW**) representation of each document. And also a Naive Bayes (**NB**) classifier.

We split the data, so that **20%** of them remain for testing.

__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.95      0.99      0.97       575
      likes       0.99      0.95      0.97       621

avg / total       0.97      0.97      0.97      1196
```

#### Experiment 2: TF - NB - 20% test

In this experiment we use a Term Frequency (**TF**) representation of each document. And also a Naive Bayes (**NB**) classifier.

We split the data, so that **20%** of them remain for testing.

__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.97      0.92      0.94       633
      likes       0.91      0.97      0.94       563

avg / total       0.94      0.94      0.94      1196
```

#### Experiment 3: TFIDF - NB - 20% test

In this experiment we use a **TFIDF** representation of each document. And also a Naive Bayes (**NB**) classifier.

We split the data, so that **20%** of them remain for testing.

__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.96      0.95      0.95       584
      likes       0.95      0.96      0.96       612

avg / total       0.95      0.95      0.95      1196
```

#### Experiment 4: TFIDF - SVM - 20% test

In this experiment we use a **TFIDF** representation of each document. And also a linear Support Vector Machine (**SVM**) classifier.

We split the data, so that **20%** of them remain for testing.

__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.96      0.97      0.97       587
      likes       0.97      0.96      0.97       609

avg / total       0.97      0.97      0.97      1196
```

#### Experiment 5: TFIDF - SVM - KFOLD

In this experiment we use a **TFIDF** representation of each document. And also a linear Support Vector Machine (**SVM**) classifier.

We split the data using Stratified **K-Fold** algorithm with **k = 5**.

__Results__:

```
Mean accuracy: 0.977 (+/- 0.002 std)
```

#### Experiment 5: BOW - NB - KFOLD

In this experiment we use a **TFIDF** representation of each document. And also a linear Support Vector Machine (**SVM**) classifier.

We split the data using Stratified **K-Fold** algorithm with **k = 5**.

__Results__:

```
Mean accuracy: 0.968 (+/- 0.002 std)
```

#### Experiment 6: TFIDF - SVM - 90% test

In this experiment we use a **TFIDF** representation of each document. And also a linear Support Vector Machine (**SVM**) classifier.

We split the data, so that **90%** of them remain for testing! Only 10% of the dataset is used for training!

__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.90      0.95      0.93      2689
      likes       0.95      0.90      0.92      2693

avg / total       0.92      0.92      0.92      5382
```

#### Experiment 7: TFIDF - SVM - KFOLD - 20 classes

In this experiment we use a **TFIDF** representation of each document. And also a linear Support Vector Machine (**SVM**) classifier.

We split the data using Stratified **K-Fold** algorithm with **k = 5**.

We also use the whole "Twenty Newsgroups" dataset, which has **20** classes.

__Results__:

```
Mean accuracy: 0.892 (+/- 0.001 std)
```

#### Experiment 7: BOW - NB - KFOLD - 20 classes

In this experiment we use a Bag Of Words (**BOW**) representation of each document. And also a Naive Bayes (**NB**) classifier.

We split the data using Stratified **K-Fold** algorithm with **k = 5**.

We also use the whole "Twenty Newsgroups" dataset, which has **20** classes.

__Results__:

```
Mean accuracy: 0.839 (+/- 0.003 std)
```

#### Experiment 8: TFIDF - 5-NN - Distance Weights - 20% test

In this experiment we use a **TFIDF** representation of each document. And also a K Nearest Neighbors (**KNN**) classifier with **k = 5** and **distance weights**.

We split the data using Stratified **K-Fold** algorithm with **k = 5**.

__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.93      0.88      0.90       608
      likes       0.88      0.93      0.90       588

avg / total       0.90      0.90      0.90      1196
```

#### Experiment 9: TFIDF - 5-NN - Uniform Weights - 20% test

In this experiment we use a **TFIDF** representation of each document. And also a K Nearest Neighbors (**KNN**) classifier with **k = 5** and **uniform weights**.

We split the data using Stratified **K-Fold** algorithm with **k = 5**.

__Results__:

```
             precision    recall  f1-score   support

   dislikes       0.95      0.90      0.92       581
      likes       0.91      0.95      0.93       615

avg / total       0.93      0.93      0.93      1196
```

#### Experiment 10: TFIDF - 5-NN - Distance Weights - KFOLD

In this experiment we use a **TFIDF** representation of each document. And also a K Nearest Neighbors (**KNN**) classifier with **k = 5** and **distance weights**.

We split the data using Stratified **K-Fold** algorithm with **k = 5**.

__Results__:

```
Mean accuracy: 0.908 (+/- 0.003 std)
```

#### Experiment 11: TFIDF - 5-NN - Distance Weights - KFOLD - 20 classes

In this experiment we use a **TFIDF** representation of each document. And also a K Nearest Neighbors (**KNN**) classifier with **k = 5** and **distance weights**.

We split the data using Stratified **K-Fold** algorithm with **k = 5**.

We also use the whole "Twenty Newsgroups" dataset, which has **20** classes.

__Results__:

```
 Mean accuracy: 0.745 (+/- 0.002 std) 
```

### So What?

This experiments show that text classification can be effectively done by simple tools like TFIDF and SVM.

#### Any Conclusion?

We have found that TFIDF with SVM have the best performance.

TFIDF with SVM perform well both for 2-class problem and 20-class problem.

I would say if you want suggestion from me, use **TFIDF with SVM**.