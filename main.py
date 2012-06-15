from colorama import init
from termcolor import colored
import sklearn.datasets
import sklearn.feature_extraction.text
from sklearn.naive_bayes import MultinomialNB
def main():
	init()


	test_main()

	exit();
	# Load All the files
	files = sklearn.datasets.load_files('dataset', shuffle=True)
	
	# BagofWords
	count_vector = sklearn.feature_extraction.text.CountVectorizer()
	
	
	# num = []
	# for i in range(len(files.filenames)):
	# 	try:
	# 		X_counts = count_vector.fit_transform(files.data[i:i+1])
	# 	except Exception, e:
	# 		# print i, files.filenames[i]; exit()
	# 		num.append(files.filenames[i])
	# print num;exit()

	X_counts = count_vector.fit_transform(files.data)
	
	tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(X_counts)
	X_tf = tf_transformer.transform(X_counts)
	

	clf = MultinomialNB().fit(X_tf, files.target)

	docs_new = ['I want to sell this low price stuff', 'Playing with new skates and stick']
	X_new_counts = count_vector.transform(docs_new)
	X_new_tf = tf_transformer.transform(X_new_counts)

	predicted = clf.predict(X_new_tf)

	for doc, cat in zip(docs_new, predicted):
		print '%r => %s' % (doc, files.target_names[cat])


	# add cross validation for testing
	# add naive bayes classifier and BOW
	# add better classifier with TFIDF

def test_main():
	directory = 'ds2'
	files = sklearn.datasets.load_files(directory)
	files.data = refine_all_emails(files.data)
	print files.data[0]

"""
Does `refine_single_email` for every single email included in the list
parameter is a list of strings
returns a list of strings
"""
def refine_all_emails(file_data):
	newfiles = []
	for email in file_data:
		newfiles.append(refine_single_email(email))
	return newfiles

"""
Delete the unnecessary information in the header of emails
Deletes only lines in the email that starts with 'Path:', 'Newsgroups:', 'Xref:'
parameter is a string.
returns a string.
"""
def refine_single_email(email):	
	parts = email.split('\n')
	newparts = []

	finished = False
	for part in parts:
		if finished:
			newparts.append(part)
			pass
		if not (part.startswith('Path:') or part.startswith('Newsgroups:') or part.startswith('Xref:')) and not finished:
			newparts.append(part)
		if part.startswith('Lines:'):
			finished = True

	return '\n'.join(newparts)

if __name__ == '__main__':
	main()