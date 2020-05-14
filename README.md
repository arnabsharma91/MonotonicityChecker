# ISSTA2020 Experiment Replication Package for Testing Monotonicity
Our paper [Higher Income, Larger Loan?Monotonicity Testing of Machine Learning Models](https://conf.researchr.org/track/issta-2020/issta-2020-papers#event-overview) is published in ISSTA 2020. This page contains the replication package to repeat the experiments described in the paper.


## Contributors of the paper
[Heike Wehrheim](https://cs.uni-paderborn.de/index.php?id=70604) & 
[Arnab Sharma](https://cs.uni-paderborn.de/index.php?id=67148)


## Installing required packages
To install the required software to run our experiments you need to first open the terminal inside this directory and run the following command:

$ pip install -r requirements.txt

This will install all the required packages and software in the current directory automatically.

These steps should install all the required packages to run our code. However, if you still face any problem, you can download any package by using 'pip':

$ pip install <package>
 
 ## Reproducing the results

This repository contains the necessary files and datasets to replicate the results of our paper .

We have used python version 3.6.5. Hence, we request the user to use this python version while using our package. 

We have performed all of our experiments in Windows. However, we have also created a version which can be run in Linux. If you want run our code in Linux use MonotonicityArtifact4Linux.

To run our packages without any problem we strongly recommend you to use anaconda. 

This folder contains the following:

AdaptiveRandomTesting/	  		-- This folder contains the code and datasets needed to replicate the results for Adaptive Random Testing 						technique

PropertyBasedTestingRQ1/	  	-- This folder contains the code and datasets needed to replicate the results for Property based Testing 						               technique for research question 1 in the paper. As the settings of quickcheck (property based testing 								tool) are different for two research questions we have 2 seperate folder for property based 								testing.

PropertyBasedTestingRQ2/		-- This folder contains the code and datasets needed to replicate the results for Property based Testing 						               technique for research question 2 in the paper.


Pruning_Analysis/			-- This folder contains the code and datasets needed to replicate the results for verification based Testing 						               technique for research question 4 in the paper.

VerificationBasedTesting		-- This folder contains the code and datasets needed to replicate the results for verification based Testing 						               technique for research questions 1,2,3 in the paper.

IMP: Each of these folder contains a README_LOCAL.txt file which will guide you to run the script for corresponding technique
 

The scripts will generate some intermediate files and datasets. In the end, the outputs will be stored in the Output folder. 
While running 
LightGbm algorithm some deprecation warnings might occur. Also, you might see some convergence problems with some ML algorithms.
This is expected.

When running the script it will ask you for some inputs like: 1)MAX_SAMPLES limit and 2)No of times test cases to run. We have done our experiments by setting MAX_SAMPLES to 1000 and run each test cases at least 10 times. We suggest you to do the same as each of our technique involves some sort of randomness.

Please note that our approach involves lot of randomness. We try to avoid that by setting the input parameters to fixed values and running our test cases over and over again. Even after that we observed some randomness still persist. 

All the execution times written in output files are in seconds.



