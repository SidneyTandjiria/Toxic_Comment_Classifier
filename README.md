# COMP9417 Machine Learning Project: Toxic Comment Classification

Names: Sidney Tandjiria, Xinli Wang

## Link to the Kaggle competition

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

## Files submitted

1. **fit_models.py**: Main python script that fits our models and produces the files located in the results folder. 

Our script was built using Python 3.8 (however, it also runs on CSE VLAB which uses Python 3.7). Running this script successfully produces 10 csv files (see results below), with main result tables also printed to terminal as the script progresses. Note that the run we used to produce our final results took 4 hours. A fast run can be configured to see our script in action using a smaller subsample of the data and reduced cross-validation loops (see the section titled 'How to do a fast run of fit_models.py').

All the modules used in the script are fairly standard - pandas, sklearn, time, collections, copy and pprint.

2. **results**: A folder containing the output csv files from running fit_models.py. These results are referenced in our report. Note that rerunning fit_models.py will regenerate these csv files.
    * training_data_summary.csv
    * sampled_training_data_summary.csv
    * 0_outer_cross_validation_results.csv
    * 1_inner_cross_validation_results_severe_toxic.csv
    * 1_inner_cross_validation_results_threat.csv
    * 1_inner_cross_validation_results_toxic.csv
    * 1_inner_cross_validation_results_identity_hate.csv
    * 1_inner_cross_validation_results_insult.csv
    * 1_inner_cross_validation_results_obscene.csv
    * 2_model_scores.csv

3. **report.pdf**: Our report.

4. **hyperparm_exploration.py**: Contains preliminary testing of the effect of certain hyperparameters tested over a range of values. Results from this are refered to in the appendix of our report.

## Location of datasets

The datasets used by fit_models.py are not included in our submission due to their large size. They can be found in the following link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

In order to run fit_models.py, all four of the data files from Kaggle must be downloaded, unzipped and placed in a folder named **data** located in the same location as the fit_models.py script:
* train.csv
* test.csv
* test_labels.csv
* sample_submission.csv (this is only used as a template to generate a submission file for kaggle)

## How to do a fast run of fit_models.py

Due to the size of the dataset and our use of nested cross validation, running the script as-is to get the results in our report took roughly 4 hours. 

To see the script in action (albeit much faster!), there are a few parameters which can be changed by the user (all of these changes can be made in the 'main' section of the script, with comments to help identify them):
   * sample_size can be reduced to 0.1 (originally 0.5) - reduces the amount of training data
   * inner_cv_split can be reduced to 2 (originally 3) - reduces the number of inner cv loops
   * outer_cv_split can be reduced to 3 (originally 5) - reduces the number of outer cv loops
   * n_iter can be reduced to 5 (originally 10) - reduces the sets of hyperparameters considered by RandomizedSearchCV
   
The above settings took about 10 minutes to run on VLAB, and produced a final test ROC AUC of 0.949 (not bad, especially compared to our 0.966 with 50% of the data which took 4 hours to run!). You could potentially speed up the run further by reducing the above parameters even more, but it will probably come with reduced performance.

## Kaggle submission

We didn't include the final submission.csv file (which can be submitted to Kaggle) in our project submission as this was very large - this section is commented out in our script. If you'd like to see it, uncomment the last few lines of our script that produces this file, and do a fast run as described above. This will produce a submission.csv file which can be submitted to Kaggle.
