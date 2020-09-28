# Sparkify
This is a capstone Spark project for Udacity Data Science program whose mission is to predict churn for the imaginary music streaming service called *Sparkify* which emulates a real streaming service like Spotify.

## Motivation
I found very attractive the idea to address big data in the Udacity's capstone project. The project data comes with the small sample dataset available on local machine and with the big dataset of 12GB available on Amazon EMR cluster. I have performed data exploratory analysis, feature engineering, and modelling with both datasets, however, the small dataset was ideal to get to know better the data while the full dataset was more suitable for modelling and model fine tuning given that *the more data, the better*.  

## Project Overview
Business-wise, the word "churn" is used to refer to customers that leave the company’s service over a given time period. The businesses strive to identify the potential users who are likely to leave before they actually leave the service in order to take actions to retain them. Some estimate that it may costs five or six times as much to acquire a new customer than it does to retain the ones you have (https://baremetrics.com/academy/churn).

The Sparkify users can stream the music using the free subscription plan with ads or paid subscription plan without ads. Apart from using the service to listen to the music, users can thumb up or down, add songs to playlists, or add friends. Users are free to change their subscription plan by *upgrading* from free to paid, by *downgrading* from paid to free, or to entirely stop using the service by *cancelling* the subscription. Our churned users are defined as those who *downgrade* or *cancel* the subscription. In our project, we separately address these two churn types of users - **cancelled** and **downgraded** users.

The project goal is to provide a machine learning model that can successfully identify churned users of each churn type according to the chosen metrics and expected results. 

## Results
We have achieved **0.8994** of F1-score on cluster and **0.9284** of F1-score on sample dataset, both for *cancelled users*. The result for *downgraded* users - only available on cluster - is **0.8268** of F1-score. Please check the [main project notebook](https://github.com/amosvoron/sparkify/blob/master/Sparkify.ipynb) and the medium article [Churn prediction with Sparkify](https://medium.com/@amos.voron/churn-prediction-with-sparkify-6f9127da7235) for project details.

## Installation
### Clone
```sh
$ git clone https://github.com/amosvoron/sparkify.git
```

## Libraries
- Spark 2.4.5

## Repository Description

```sh
- Sparkify-cluster-cancelled-bestmodel.py                  # python script for cancelled users best model fitting on cluster
- Sparkify-cluster-cancelled-feature-engineering.ipynb     # notebook with feature engineering code for cancelled users
- Sparkify-cluster-downgraded-bestmodel.py                 # python script for downgraded users best model fitting on cluster
- Sparkify-cluster-downgraded-feature-engineering.ipynb    # feature engineering code for downgraded users
- Sparkify-cluster-results.ipynb                           # fitting results on cluster
- Sparkify-modelling.ipynb                                 # modelling code for cancelled users (sample dataset)
- Sparkify.ipynb                                           # main project notebook
- README.md                                                # README file
- LICENCE.md                                               # LICENCE file
```

## Acknowledgements
Thanks to Udacity for the unique "big data experience" with spark and to Amazon for helping me with the AWS credit to execute exhaustive fitting jobs on cluster. 

## License

MIT