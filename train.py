from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, recall_score, fbeta_score
from scipy.sparse import hstack
import pandas as pd
import time
import sys
import os
import joblib 

# Load data


# Score function
def scores(labels, predictions):
    f1 = round(f1_score(labels, predictions), 4) * 100
    roc = round(roc_auc_score(labels, predictions), 4) * 100
    brier_loss = round(brier_score_loss(labels, predictions), 4)
    brier = (1 - brier_loss) * 100
    c1 = round(recall_score(labels, predictions, average='macro'), 4) * 100
    f05 = round(fbeta_score(labels, predictions, beta=0.5), 4) * 100

    score_dict = {'F1': f1, 'Roc_auc': roc, 'Brier': brier, 'C@1': c1, 'F_05': f05}
    score_dict['Mean'] = sum(score_dict.values()) / len(score_dict)
    return score_dict

def train_svm_combined(train, val, m, n, word_feature_limit, punct_feature_limit):
    word_token_pattern = r"(?u)\b\w+\b"  # standard word token pattern
    punct_token_pattern = r"[\!\?\,\.\'\"\`\:\;]+"  # just punctuation
    # Vectorizer for n-grams
    vectorizer_words = CountVectorizer(ngram_range=(m, n), max_features=word_feature_limit, token_pattern=word_token_pattern)

    # Vectorizer for punctuation
    vectorizer_punct = CountVectorizer(max_features=punct_feature_limit, token_pattern=punct_token_pattern)

    start_time = time.time()

    # Fit on training data
    x_train_words = vectorizer_words.fit_transform(train['text'])
    x_test_words = vectorizer_words.transform(val['text'])

    x_train_punct = vectorizer_punct.fit_transform(train['text'])
    x_test_punct = vectorizer_punct.transform(val['text'])

    # Combine features
    x_train_combined = hstack([x_train_words, x_train_punct])
    x_test_combined = hstack([x_test_words, x_test_punct])

    y_train = train['label']
    y_test = val['label']

    end_time = time.time()
    print("Vectorization took {} seconds".format(round(end_time - start_time, 2)))

    # Train SVC 
    start_time = time.time()
    model = SVC()
    model.fit(x_train_combined, y_train)
    print("Training SVC took {} seconds".format(round(time.time() - start_time, 2)))

    # Predict
    predictions = model.predict(x_test_combined)
    val = val.drop(columns=["text", "label", "genre", "model"])
    val['label'] = predictions

    modelfile = 'model_and_vectorizers.joblib'

    joblib.dump((model, vectorizer_words, vectorizer_punct), modelfile)

    print(f"Model saved as {modelfile}!")

    return val

if __name__ == "__main__":
    if len(sys.argv)!= 3:
        print("Usage: python script.py <input_directory> <output_directory")
        sys.exit(1)
    
    inputdir = sys.argv[1]
    outputdir = sys.argv[2]
    try:
        for file in os.listdir(inputdir):
            if file.startswith('train') and file.endswith('.jsonl'):
                file_path = os.path.join(inputdir, file)
                train = pd.read_json(file_path, lines=True)
            if file.startswith('val') and file.endswith('.jsonl'):
                file_path = os.path.join(inputdir, file)
                val = pd.read_json(file_path, lines=True)
        val = train_svm_combined(train, val, 1, 2, 40, 15)
    except:
       print('Problem with input directory!')
    try:
        val.to_json(outputdir+'/Results.jsonl', lines=True, orient='records')
        print('Results have been saved!')
    except:
        print('Problem with output directory')