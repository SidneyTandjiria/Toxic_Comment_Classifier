import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from time import time
from collections import defaultdict
import copy
from pprint import pprint

# Run on Python 3.8.0 (Windows)
# Also tested and runs successfully on CSE VLAB (Python 3.7, Linux)

# Create a class for comparing different learning algorithms
class PipelineComparison:
    
    def __init__(self, pipelines, pipeline_params):
        self.pipelines = pipelines # dict of pipelines
        self.pipeline_params = pipeline_params # dict of parameters for each pipeline
        self.pipeline_names = pipelines.keys() # name of pipelines
        self.grid_searches = {} # store grid searches
        self.pipeline_outer_score = defaultdict(dict) # outer cv scores
        self.summary = pd.DataFrame() # outer CV summary
        self.best_pipelines = {} # store the best pipeline for each label
        self.best_pipelines_names = {} # stores the name of the best pipeline for each label
    
    # Performs a nested cross validated grid search
    # Loops through each pipeline/parameter set, for each label
    def fit(self, X, y, y_labels, inner_cv_split = 3, outer_cv_split = 5, n_iter = 10, scoring = 'roc_auc', verbose = 1):
        
        inner_cv = StratifiedKFold(n_splits = inner_cv_split, shuffle = True, random_state = 3)
        outer_cv = StratifiedKFold(n_splits = outer_cv_split, shuffle = True, random_state = 9)

        # Loop through each of the pipelines
        print("\nStarting the nested cross validation...") 
        for pipe in self.pipeline_names:
            pipeline = self.pipelines[pipe]
            pipeline_param = self.pipeline_params[pipe]
            # For each pipeline, do a grid search on the hyperparameters
            grid_search = RandomizedSearchCV( # Picks a random set of hyperparameters, as opposed to GridSearchCV
                pipeline,
                pipeline_param,
                n_iter = n_iter, # number of parameter settings that are sampled
                cv = inner_cv, # k-fold cross validation
                scoring = scoring, # scoring metric
                verbose = verbose, # make this bigger for more comprehensive log
                n_jobs = 6, # use all available processors in parallel (faster)
                refit = True # refit the model on the best parameters
            )
            # grid_search = GridSearchCV( # exhaustively tries every combination of hyperparameters, can be slow with too many parameters
            #     pipeline,
            #     pipeline_param,
            #     cv = inner_cv, # k-fold cross validation
            #     scoring = scoring, # scoring metric
            #     verbose = verbose, # make this bigger for more comprehensive log
            #     n_jobs = -1, # use all available processors in parallel (faster)
            #     refit = True # refit the model on the best parameters
            # )
            self.grid_searches[pipe] = copy.deepcopy(grid_search)
            for label in y_labels:
                # This is the outer cv loop
                # It gives us an estimation of generalisation error of the algorithm/pipeline
                # We don't need to know the hyperparameters at this stage, we only care about the process for picking the hyperparameters
                # If our scores here are very different, then we know the algorithm is unstable
                nested_cv_scores = cross_val_score(
                    grid_search,
                    X = X,
                    y = y[label],
                    cv = outer_cv,
                    n_jobs = 6 # use all available processors in parallel (faster)
                    )
                self.pipeline_outer_score[pipe][label] = nested_cv_scores # list of cv scores
        
        # Summarise the outer CV results
        self.summary = self.summarise() # summarises and saves the outer cv results in a csv file
        print("\nOuter cross validation results for each pipeline...\n")
        print(self.summary)

        # Train on the best pipeline per label
        print("\nFitting the best pipeline and finding optimal hyperparameters for each label...\n")
        self.fit_best_pipelines(X, y) # fits 6 models, saves results of inner cv to 6 csv files

    # we will fit the best pipeline per label on the entire training set
    def fit_best_pipelines(self, X, y):
        best_pipelines = (self.summary.groupby('label').apply(lambda x: x.sort_values('score_mean', ascending=False).head(1))[['label', 'pipeline']])
        pprint(best_pipelines)
        best_pipelines_list = best_pipelines.values.tolist()
        for p in best_pipelines_list:
            label = p[0]
            pipeline = p[1]
            gs = self.grid_searches[pipeline]
            gs.fit(X, y[label]) # fit on all training data!
            self.best_pipelines[label] = copy.deepcopy(gs) # save the fit
            self.best_pipelines_names[label] = pipeline # save the name of the pipeline
            cv_results = pd.DataFrame(gs.cv_results_)
            cv_results.to_csv(f"1_inner_cross_validation_results_{label}.csv", index=True)
            print(f"\n{label}: ")
            print(cv_results)       
        
    # Use the best pipelines for each label to get a probability for each label
    # Returns a dataframe with 6 columns (one probability per label)
    def predict_proba(self, X): # feed it a dataset to make predictions on
        p = {}
        for label in self.best_pipelines:
            gs = self.best_pipelines[label]
            probabilities = gs.predict_proba(X)[:,1]
            p[label] = probabilities
        probs = pd.DataFrame(p)
        return probs

    def score(self, X_train, y_train, X_test, y_test): # feed it a dataset and labels and give back the AUC ROC summary
        print('\nScoring training and test datasets...')
        results = []
        for label in self.best_pipelines:
            gs = self.best_pipelines[label]
            pipe = self.best_pipelines_names[label] # name of best estimator
            params = gs.best_params_ # best parameters
            cv_score = gs.best_score_ # from cross validation
            train_score = gs.score(X_train, y_train[label]) # this should be more optimistic than the CV score (since more data is used!)
            test_score = gs.score(X_test, y_test[label]) # for generalising error
            row = pd.Series({
                **{"pipeline": pipe},
                **{"best_params": params},
                **{"label": label},
                **{"cv_score": cv_score},
                **{"training_score": train_score},
                **{"test_score": test_score},
            })
            results.append(row)
        score_summary = pd.concat(results, axis = 1).T
        print("\nSummary of model ROC AUC:")
        print(score_summary)
        print(f"\nMean test ROC AUC: {score_summary['test_score'].mean()}")
        score_summary.to_csv("2_model_scores.csv", index=True)

    def summarise(self):
        results = []
        for pipe in self.pipeline_outer_score:
            for label in self.pipeline_outer_score[pipe]:
                scores = self.pipeline_outer_score[pipe][label]
                result = pd.Series({
                    **{"pipeline": pipe},
                    **{"label": label},
                    **{"score_mean": scores.mean()}, # mean of scores
                    **{"score_sd": scores.std()}, # standard deviation of scores
                    **{f"score_{i}": scores[i] for i in range(len(scores))} # all scores
                })
                results.append(result)
        summary = pd.concat(results, axis = 1).T
        common_columns = ["label", "pipeline", "score_mean", "score_sd"]
        column_order = common_columns + [col for col in summary.columns if col not in common_columns]
        summary = summary[column_order]
        summary.to_csv("0_outer_cross_validation_results.csv", index=True)
        return summary

if __name__ == "__main__":

    t0 = time()

    # Read in data (data is stored in a folder named 'data')
    train = pd.read_csv('data/train.csv')
    raw_test = pd.read_csv('data/test.csv')
    raw_test_labels = pd.read_csv('data/test_labels.csv')
    sub = pd.read_csv('data/sample_submission.csv') # only used as a template for creating a kaggle submission file

    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    train.describe().to_csv("training_data_summary.csv", index=True)

    # Filter out unlabelled test data
    # Kaggle scores on a sample of the test dataset, so we don't have labels on some of these
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
        df.cleaned_comment.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True) # remove non-ascii characters
        return(df)

    # Clean the data
    print("\nCleaning the data...")
    
    train_cleaned = clean_text(train)
    test_cleaned = clean_text(test)
    raw_test_cleaned = clean_text(raw_test)

    # Dataset is quite large so we will take a sample (50%) from training
    sample_size = 0.5 
    #sample_size = 0.1 # use this for a fast run

    X_train, _, y_train, _ = train_test_split(
        train_cleaned['cleaned_comment'], # text
        train_cleaned[train.columns & labels], # different labels
        test_size = 1-sample_size, 
        random_state = 0
    )

    combined_train = pd.concat([X_train, y_train], axis = 1)
    combined_train.describe().to_csv("sampled_training_data_summary.csv", index=True)

    ############## ADD IN PIPELINES AND PARAMETERS FOR EACH PIPELINE HERE

    # common hyperparameter ranges/distributions
    cnt_vect__max_features = [None, 1000, 10000, 50000] # maximum number of features
    cnt_vect__ngram_range = [(1,1), (1,2), (2,2)] # unigrams and/or bigrams
    cnt_vect__binary = [True, False] # occurence of word if True, else count of words if False
    cnt_vect__stop_words = ['english', None] # remove stop words or not
    cnt_vect__lowercase = [True, False]
    tf_idf__use_idf = [False, True] # use tf or tf-idf
    tf_idf__norm = ['l1', 'l2'] # l1 or l2 norm

    pipe_MNB_tfidf = Pipeline([
        ('cnt_vect', CountVectorizer()),
        ('tf_idf', TfidfTransformer()),
        ('MNB', MultinomialNB())
    ])

    param_MNB_tfidf = {
        'cnt_vect__max_features': cnt_vect__max_features # maximum number of features
        , 'cnt_vect__ngram_range': cnt_vect__ngram_range # unigrams or bigrams
        , 'cnt_vect__binary': cnt_vect__binary
        , 'cnt_vect__stop_words': cnt_vect__stop_words
        , 'cnt_vect__lowercase': cnt_vect__lowercase
        , 'tf_idf__use_idf': tf_idf__use_idf # use tf or tf-idf
        , 'tf_idf__norm': tf_idf__norm # l1 or l2 norm
        , 'MNB__alpha': [0.01, 0.1, 0.5, 1] # smoothing parameter, 1 is laplace smoothing
        , 'MNB__fit_prior': [True, False] # assume uniform prior if false
    }

    # Logistic Regression
    pipe_logreg_tfidf = Pipeline([
        ('cnt_vect', CountVectorizer()),
        ('tf_idf', TfidfTransformer()),
        ('LogisticRegression', LogisticRegression())
    ])

    param_logreg_tfidf = {
         'cnt_vect__max_features': cnt_vect__max_features # maximum number of features
        , 'cnt_vect__ngram_range': cnt_vect__ngram_range # unigrams or bigrams
        , 'cnt_vect__binary': cnt_vect__binary
        , 'cnt_vect__stop_words': cnt_vect__stop_words
        , 'cnt_vect__lowercase': cnt_vect__lowercase
        , 'tf_idf__use_idf': tf_idf__use_idf # use tf or tf-idf
        , 'tf_idf__norm': tf_idf__norm # l1 or l2 norm
        , 'LogisticRegression__class_weight': [None, 'balanced']
        , 'LogisticRegression__C': [1, 2, 3, 4, 5]
    }

    # Support Vector Machine
    pipe_SVM_tfidf = Pipeline([
        ('cnt_vect', CountVectorizer()),
        ('tf_idf', TfidfTransformer()),
        ('SVM', SGDClassifier())
    ]) 

    param_SVM_tfidf = {
         'cnt_vect__max_features': cnt_vect__max_features # maximum number of features
        , 'cnt_vect__ngram_range': cnt_vect__ngram_range # unigrams or bigrams
        , 'cnt_vect__binary': cnt_vect__binary
        , 'cnt_vect__stop_words': cnt_vect__stop_words
        , 'cnt_vect__lowercase': cnt_vect__lowercase
        , 'tf_idf__use_idf': tf_idf__use_idf # use tf or tf-idf
        , 'tf_idf__norm': tf_idf__norm # l1 or l2 norm
        , 'SVM__penalty': ['l1', 'l2', 'elasticnet']
        , 'SVM__class_weight': [None, 'balanced']
        , 'SVM__alpha': [0.0001, 0.001, 0.01, 0.1]
    }

    pipeline_input = dict(
        pipe_MNB_tfidf = pipe_MNB_tfidf,
        pipe_SVM_tfidf = pipe_SVM_tfidf,
        pipe_logreg_tfidf = pipe_logreg_tfidf
    )

    parameter_input = dict(
        pipe_MNB_tfidf = param_MNB_tfidf,
        pipe_SVM_tfidf = param_SVM_tfidf,
        pipe_logreg_tfidf = param_logreg_tfidf
    )
    
    ################## GET RESULTS

    pc = PipelineComparison(pipeline_input, parameter_input)

    pc.fit(X_train, y_train, labels, inner_cv_split = 3, outer_cv_split = 5, n_iter = 10, verbose = 0) # does nested CV to fit the best models for each label
    #pc.fit(X_train, y_train, labels, inner_cv_split = 2, outer_cv_split = 3, n_iter = 5, verbose = 0) # Use this line instead for a fast run

    pc.score(X_train, y_train, test_cleaned['cleaned_comment'], test_labels) # provides a summary of results

    print("\nModel training and evaluation completed in %0.2f minutes." %((time() - t0)/60))

    ################## Uncomment below to produce a csv file for Kaggle submission

    # submission = pc.predict_proba(raw_test_cleaned['cleaned_comment'])
    # submission['id'] = sub['id'] # add on id column
    # submission = submission[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    # submission.to_csv("submission.csv", index=False)