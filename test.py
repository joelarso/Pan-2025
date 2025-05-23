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
	    print("Usage: python3 test.py <input_directory> <output_directory>")
	    print(len(sys.argv))
	    sys.exit(1)
	

	
	inputfile = sys.argv[1]
	outputdir = sys.argv[2]

	print(inputfile)
	print(outputdir)

	try:
		print("Checking input file existence:", os.path.exists(inputfile))
		print("Absolute path to input file:", os.path.abspath(inputfile))
		dataset = pd.read_json(inputfile, lines=True)
		print('Dataset has been loaded!')
		modelfile = 'model_and_vectorizers.joblib'
		prediction_data = make_predictions(dataset, modelfile)
	except:
		print('Problem with input file!')
	try: 
		resultfile = 'predictions.jsonl'
		outputdir = os.path.abspath(outputdir)
		os.makedirs(outputdir, exist_ok=True)
		file_path = os.path.join(outputdir, resultfile)
		print(f"Attempting to save predictions to: {file_path}")
		prediction_data.to_json(file_path, lines=True, orient='records')
		print('Results have been saved!')
	except Exception as e:
		print(f'Problem with output directory {e}!')