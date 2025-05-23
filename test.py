from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from scipy.sparse import hstack
import pandas as pd
import time
import sys
import os
import joblib


def make_predictions(dataset, modelfile):
	
	m = 1
	n = 2
	word_feature_limit = 40
	punct_feature_limit = 15

	word_token_pattern = r"(?u)\b\w+\b"  # standard word token pattern
	punct_token_pattern = r"[\!\?\,\.\'\"\`\:\;]+"  # just punctuation
	
	model, vectorizer_words, vectorizer_punct = joblib.load(modelfile)

	x_test_words = vectorizer_words.transform(dataset['text'])
	x_test_punct = vectorizer_punct.transform(dataset['text'])
	x_test_combined = hstack([x_test_words, x_test_punct])

	predictions = model.predict(x_test_combined)
	dataset = dataset.drop(columns=["text", "label"])
	dataset["label"] = predictions
	
	return dataset

if __name__ == "__main__":
	if len(sys.argv)!= 3:
	    print("Usage: python3 test.py <input_directory> <output_directory")
	    sys.exit(1)
	
	inputfile = sys.argv[1]
	outputdir = sys.argv[2]

	try:
		dataset = pd.read_json(inputfile, lines=True)
		modelfile = 'model_and_vectorizers.joblib'
		prediction_data = make_predictions(dataset, modelfile)
	except:
		print('Problem with input file!')
	try: 
		resultfile = 'results.jsonl'
		prediction_data.to_json(os.path.join(outputdir, resultfile), lines=True, orient='records')
		print('Results have been saved!')
	except:
		print('Problem with output directory!')