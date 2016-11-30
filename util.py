from colorama import init
from termcolor import colored
import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import sklearn.cross_validation
import sklearn.svm
import sklearn.neighbors


def main():
    from colorama import init
    from termcolor import colored
    init()

    test_main()


def test_main():
    directory = 'ds2'
    directory = 'dataset'
    directory = 'ds3'
    # load the dataset from disk
    files = sklearn.datasets.load_files(directory)

    # refine them
    refine_all_emails(files.data)

    # calculate the BOW representation
    word_counts = bagOfWords(files.data)

    # TFIDF
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X_tfidf = tf_transformer.transform(word_counts)

    X = X_tfidf

    # cross validation
    # clf = sklearn.naive_bayes.MultinomialNB()
    # clf = sklearn.svm.LinearSVC()
    n_neighbors = 5
    weights = 'uniform'
    # weights = 'distance'
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    scores = cross_validation(X, files.target, clf, cv=5)
    pretty_print_scores(scores)


def delete_incompatible_files(files):
    """
    Deletes the list of files that are passed to it from file system!
    argument `files` is a list of strings. containing absolute or relative pathes
    """
    import os
    for f in files:
        print colored("deleting file:", 'red'), f
        os.remove(f)


def find_incompatible_files(path):
    """
    Finds the filenames that are incompatible with `CountVectorizer`. These files are usually not compatible with UTF8!
    parameter `path` is the absolute or relative path of the training data's root directory.
    returns a list of strings.
    """

    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    files = sklearn.datasets.load_files(path)
    num = []
    for i in range(len(files.filenames)):
        try:
            count_vector.fit_transform(files.data[i:i + 1])
        except UnicodeDecodeError:
            num.append(files.filenames[i])
        except ValueError:
            pass

    return num


def pretty_print_scores(scores):
    """
    Prints mean and std of a list of scores, pretty and colorful!
    parameter `scores` is a list of numbers.
    """
    print colored("                                      ", 'white', 'on_white')
    print colored(" Mean accuracy: %0.3f (+/- %0.3f std) " % (scores.mean(), scores.std() / 2), 'magenta', 'on_white', attrs=['bold'])
    print colored("                                      ", 'white', 'on_white')


def cross_validation(data, target, classifier, cv=5):
    """
    Does a cross validation with the classifier
    parameters:
        - `data`: array-like, shape=[n_samples, n_features]
            Training vectors
        - `target`: array-like, shape=[n_samples]
            Target values for corresponding training vectors
        - `classifier`: A classifier from the scikit-learn family would work!
        - `cv`: number of times to do the cross validation. (default=5)
    return a list of numbers, where the length of the list is equal to `cv` argument.
    """
    return sklearn.cross_validation.cross_val_score(classifier, data, target, cv=cv)


def bagOfWords(files_data):
    """
    Converts a list of strings (which are loaded from files) to a BOW representation of it
    parameter 'files_data' is a list of strings
    returns a `scipy.sparse.coo_matrix`
    """

    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    return count_vector.fit_transform(files_data)


def refine_all_emails(file_data):
    """
    Does `refine_single_email` for every single email included in the list
    parameter is a list of strings
    returns NOTHING!
    """

    for i, email in zip(range(len(file_data)), file_data):
        file_data[i] = refine_single_email(email)


def refine_single_email(email):
    """
    Delete the unnecessary information in the header of emails
    Deletes only lines in the email that starts with 'Path:', 'Newsgroups:', 'Xref:'
    parameter is a string.
    returns a string.
    """

    parts = email.split('\n')
    newparts = []

    # finished is when we have reached a line with something like 'Lines:' at the begining of it
    # this is because we want to only remove stuff from headers of emails
    # look at the dataset!
    finished = False
    for part in parts:
        if finished:
            newparts.append(part)
            continue
        if not (part.startswith('Path:') or part.startswith('Newsgroups:') or part.startswith('Xref:')) and not finished:
            newparts.append(part)
        if part.startswith('Lines:'):
            finished = True

    return '\n'.join(newparts)

if __name__ == '__main__':
    main()
