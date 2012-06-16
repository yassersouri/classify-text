import util
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
from colorama import init
from termcolor import colored
def main():
	# initial color printing
	init()

	dir_path = 'dataset'

	# find incompatible files
	print colored('Finding files incompatible with utf8: ', 'green', attrs=['bold'])
	incompatible_files = util.find_incompatible_files(dir_path)
	print colored(len(incompatible_files), 'yellow'), 'files found'

	# delete them
	if(len(incompatible_files) > 0):
		print colored('Deleting incompatible files', 'red', attrs=['bold'])
		util.delete_incompatible_files(incompatible_files)

	# load data
	print colored('Loading files into memory', 'green', attrs=['bold'])
	files = sklearn.datasets.load_files(dir_path)

	# refine all emails
	print colored('Refining all files', 'green', attrs=['bold'])
	util.refine_all_emails(files.data)

	# calculate the BOW representation
	print colored('Calculating BOW', 'green', attrs=['bold'])
	word_counts = util.bagOfWords(files.data)

	# TFIDF
	print colored('Calculating TFIDF', 'green', attrs=['bold'])
	tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=False).fit(word_counts)
	X = tf_transformer.transform(word_counts)

	# create classifier
	clf = sklearn.svm.LinearSVC()

	# test the classifier
	print '\n\n'
	print colored('Testing classifier with train-test split', 'blue', attrs=['bold'])
	test_classifier(X, files.target, clf, test_size=0.2, y_names=files.target_names, confusion=True)

def test_classifier(X, y, clf, test_size=0.4, y_names=None, confusion=False):
	#train-test split
	print ('test size is:', test_size)
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

	clf.fit(X_train, y_train)
	y_predicted = clf.predict(X_test)

	if not confusion:
		print colored('Classification report:', 'blue', attrs=['bold'])
		print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
	else:
		print colored('Confusion Matrix:', 'blue', attrs=['bold'])
		print sklearn.metrics.confusion_matrix(y_test, y_predicted)

if __name__ == '__main__':
	main()