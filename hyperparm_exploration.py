import pandas as pd
from time import time
import sklearn.metrics as metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Read in data
train = pd.read_csv('data/train.csv')
raw_test = pd.read_csv('data/test.csv')
raw_test_labels = pd.read_csv('data/test_labels.csv')

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Filter out unlabelled test data
index_used = raw_test_labels.index[raw_test_labels['toxic'] != -1].tolist()
test = raw_test.iloc[index_used]
test_labels = raw_test_labels.iloc[index_used]

# Function for cleaning text
def clean_text(df):
    df = df.copy(deep=True)
    df['cleaned_comment'] = df.comment_text
    # remove invalid characters from text
    df = df.replace({'cleaned_comment': r"|,|!|^|&|(|)|=|\|{|}|[|]|<|>|~|`|:|;|||\\|/|'|"}, {'cleaned_comment': ''}, regex = True)
    df = df.replace({'cleaned_comment': r'"'}, {'cleaned_comment': ''}, regex = True)
    df = df.replace({'cleaned_comment': r'\n'}, {'cleaned_comment': ' '}, regex = True)
    df = df.replace({'cleaned_comment': r'_'}, {'cleaned_comment': ' '}, regex = True) # replace underscores
    # remove non-ascii characters
    df.cleaned_comment.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
    return(df)

# Clean the data
print("Cleaning the data...")
t0 = time()
train_cleaned = clean_text(train)
test_cleaned = clean_text(test)
print("Cleaning completed in %0.3fs" % (time() - t0))

# Dataset is quite large so we will take a small sample (20%) to trail with
sample_size = 0.20
X_train, _, y_train, _ = train_test_split(
    train_cleaned['cleaned_comment'], # text
    train_cleaned[train.columns & labels], # different labels
    test_size = 1 - sample_size, random_state = 0)

inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 0)

# Multinomial Naive Bayes using counts
pipe_MNB_cnt = Pipeline([
    ('cnt_vect', CountVectorizer()),
    ('MNB', MultinomialNB())
])

param_MNB_cnt = {
    'MNB__alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # smoothing parameter, 1 is laplace smoothing
}

clf = GridSearchCV(
    estimator = pipe_MNB_cnt,
    param_grid = param_MNB_cnt,
    cv = inner_cv,
    scoring = 'roc_auc',
    verbose = 1, # make this bigger for more comprehensive log
    n_jobs = -1, # use all available processors in parallel
    refit = True # refit the model on the best parameters
)

results = pd.DataFrame({'label': [],
                        'param_MNB__alpha': [],
                        'mean_train_score': [],
                        'mean_test_score' : [],
                        'rank_test_score' : []})

#loop though each label within labels
for label in labels:
    clf.fit(X_train, y_train[label])
    temp = pd.DataFrame(clf.cv_results_)[['param_MNB__alpha', 'mean_train_score', 'mean_test_score', 'rank_test_score']]
    temp['label'] = label
    #results = results.append(pd.DataFrame(clf.cv_results_)[['param_MNB__alpha', 'mean_train_score', 'mean_test_score', 'rank_test_score']])
    results = results.append(temp)

results.to_csv("mnb_alpha_tuning.csv")

# Support Vector Machine using counts
pipe_SVM_cnt = Pipeline([
    ('cnt_vect', CountVectorizer()),
    ('SVM', SGDClassifier())
])

param_SVM_penalty = {
    'SVM__penalty': ['l1', 'l2', 'elasticnet']
}

clf = GridSearchCV(
    estimator = pipe_SVM_cnt,
    param_grid = param_SVM_penalty,
    cv = inner_cv,
    scoring = 'roc_auc',
    verbose = 1, # make this bigger for more comprehensive log
    n_jobs = -1, # use all available processors in parallel
    refit = True # refit the model on the best parameters
)

results = pd.DataFrame({'label': [],
                        'param_SVM__penalty': [],
                        'mean_train_score': [],
                        'mean_test_score' : [],
                        'rank_test_score' : []})

#loop though each label within labels
for label in labels:
    clf.fit(X_train, y_train[label])
    temp = pd.DataFrame(clf.cv_results_)[['param_SVM__penalty', 'mean_train_score', 'mean_test_score', 'rank_test_score']]
    temp['label'] = label
    results = results.append(temp)
    
results.to_csv("svm_penalty_tuning.csv")


param_SVM_cls_weight = {
    'SVM__class_weight': [None, 'balanced']
}

clf = GridSearchCV(
    estimator = pipe_SVM_cnt,
    param_grid = param_SVM_cls_weight,
    cv = inner_cv,
    scoring = 'roc_auc',
    verbose = 1, # make this bigger for more comprehensive log
    n_jobs = -1, # use all available processors in parallel
    refit = True # refit the model on the best parameters
)

results = pd.DataFrame({'label': [],
                        'param_SVM__class_weight': [],
                        'mean_train_score': [],
                        'mean_test_score' : [],
                        'rank_test_score' : []})

#loop though each label within labels
for label in labels:
    clf.fit(X_train, y_train[label])
    temp = pd.DataFrame(clf.cv_results_)[['param_SVM__class_weight', 'mean_train_score', 'mean_test_score', 'rank_test_score']]
    temp['label'] = label
    results = results.append(temp)

results.to_csv("svm_class_weight_tuning.csv")

param_SVM_alpha = {
    'SVM__alpha': [0.0001, 0.001, 0.01, 0.1]
}

clf = GridSearchCV(
    estimator = pipe_SVM_cnt,
    param_grid = param_SVM_alpha,
    cv = inner_cv,
    scoring = 'roc_auc',
    verbose = 1, # make this bigger for more comprehensive log
    n_jobs = -1, # use all available processors in parallel
    refit = True # refit the model on the best parameters
)

results = pd.DataFrame({'label': [],
                        'param_SVM__alpha': [],
                        'mean_train_score': [],
                        'mean_test_score' : [],
                        'rank_test_score' : []})

#loop though each label within labels
for label in labels:
    clf.fit(X_train, y_train[label])
    temp = pd.DataFrame(clf.cv_results_)[['param_SVM__alpha', 'mean_train_score', 'mean_test_score', 'rank_test_score']]
    temp['label'] = label
    results = results.append(temp)   

results.to_csv("svm_alpha_tuning.csv")