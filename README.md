# Sparkify
This is a capstone Spark project for Udacity Data Science program whose mission is to predict churn for the imaginary music streaming service called *Sparkify* to emulate a real streaming service like Spotify.

## Motivation
I found very attractive the idea to address big data in the Udacity's capstone project. The project data comes with the small sample dataset available on local machine and with the big dataset of 12GB available on Amazon EMR cluster. I have performed data exploratory analysis, feature engineering, and modelling with both datasets, however, the small dataset was ideal to get to know better the data while the full dataset was more suitable for modelling and model fine tuning since, as we all know, *the more data, the better*.    

## Project goal
Our principal metric is **F1-score**. The project goal is to reach (around) **0.9** of F1-score. The F1-score below **0.8** is considered as unsatisfactory.

## Results
We have achieved **0.8994** on cluster and **0.9284** on sample dataset, both for *cancelled users*. The result for *downgraded* users - only available on cluster - is **0.8268** of F1-score.

## Installation
### Clone
```sh
$ git clone https://github.com/amosvoron/sparkify.git
```

## Libraries
- Spark 2.4.5

## Repository Description

```sh
- images							                       # directory with images (used in main project notebook)
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