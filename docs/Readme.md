#This a model for failure prediction of systems based on system health data

Team members:Syreeta Fahmida, Shraddha B Surakod, M Sravani, Sadhana Hegde, Ranjitha C Gowli

We have build a set of supervised and unsupervised methods for the above purpose.  Important signals being used are memory usage, io usage, network usage and nfs usage

First we train a supervised model based on a training dataset with instance of failure and success

As part of supervised appoach we are bulding  two types of ensmbles (voting and stacking) of wide range of classifiers: 
***

The  trained model are saved in the models folder. 

At inference time, we are computing exponential moving average of the features  (these may be complicated at times). Next we load the saved models and predict failures if any.

Based on plausible reason of failure, we also try to get ot root cause and prescirbe recommendation accordlingly.

