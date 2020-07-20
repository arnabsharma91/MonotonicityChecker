# ISSTA2020 Experiment Replication Package for Testing Monotonicity
Our paper [Higher Income, Larger Loan?Monotonicity Testing of Machine Learning Models](https://conf.researchr.org/track/issta-2020/issta-2020-papers#event-overview) is published in ISSTA 2020. This page contains the replication package to repeat the experiments described in the paper.


## Contributors of the paper
[Heike Wehrheim](https://en.cs.uni-paderborn.de/sms/team/people/heike-wehrheim) & 
[Arnab Sharma](https://en.cs.uni-paderborn.de/sms/team/people/arnab-sharma)


## Installing required packages
To install the required software to run our experiments you need to first open the terminal inside this directory and run the following command:

```$ pip install -r requirements.txt```

This will install all the required packages and software in the current directory automatically.

These steps should install all the required packages to run our code. However, if you still face any problem, you can download any package by using 'pip':

```$ pip install <package>```
 
 ## Reproducing the results

This repository contains the necessary files and datasets to replicate the results of our paper .

We have used python version 3.6.5. Hence, we request the user to use this python version while using our package. 

We have performed all of our experiments in Windows. However, we have also created a version which can be run in Linux. If you want run our code in Linux use MonotonicityArtifact4Linux.

To run our packages without any problem we strongly recommend you to use anaconda. 
We have used 2 baseline approaches to compare the results with our verification-based testing approach.
If you want to check our results for adaptive random testing approach go inside the [AdaptiveRandomTesting](https://github.com/arnabsharma91/MonotonicityChecker/tree/master/AdaptiveRandomTesting) folder and follow the instructions.

We have used a property based testing tool named [quickCheck](https://pypi.org/project/pytest-quickcheck/) as our another baseline approach. We configured it in 2 ways. For the first case, we just wanted to find out, given a model whether it is monotone or not. Go [here](https://github.com/arnabsharma91/MonotonicityChecker/tree/master/PropertyBasedTestingRQ1) to check out results for this experiment. Next, we tried to look at how many test cases quickCheck needs before it finds a valid failing test cases. Check out this [folder](https://github.com/arnabsharma91/MonotonicityChecker/tree/master/PropertyBasedTestingRQ2) for replicating the results from the paper. 

We have given 2 novel strategies for generating multiple counter examples from a single one. Check out [here](https://github.com/arnabsharma91/MonotonicityChecker/tree/master/Pruning_Analysis) for finding out which strategy wins! 

Lastly, to replicate the results for our verification-based testing approach, please follow this [link](https://github.com/arnabsharma91/MonotonicityChecker/tree/master/VerificationBasedTesting).

## IMPORTANT: 
Each of these folder contains a README_LOCAL.txt file which will guide you through the steps of running the script for corresponding experiment.
 

The scripts will generate some intermediate files and datasets. In the end, the outputs will be stored in the Output folder. While running 
LightGbm algorithm some deprecation warnings might occur. Also, you might see some convergence problems with some ML algorithms.
This is expected.

When running the script it will ask you for some inputs like: 1)MAX_SAMPLES limit and 2)No of times test cases to run. We have done our experiments by setting MAX_SAMPLES to 1000 and run each test cases at least 10 times. We suggest you to do the same as each of our technique involves some sort of randomness.

Please note that our approach involves lot of randomness. We try to avoid that by setting the input parameters to fixed values and running our test cases over and over again. Even after that we observed some randomness still persist. 

All the execution times written in output files are in seconds.



