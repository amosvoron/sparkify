# Sparkify
This is a capstone Spark project for Udacity Data Science program whose mission is to predict churn for the imaginary music streaming service called *Sparkify* to emulate a real streaming service like Spotify.

## Motivation
I found very attractive the idea to address big data in the Udacity's capstone project. The project data comes with the small sample dataset available on local machine and with the big dataset of 12GB available on Amazon EMR cluster. I have performed data exploratory analysis, feature engineering, and modelling with both datasets, however, the small dataset was ideal to get to know better the data while the full dataset was more suitable for modelling and model fine tuning since, as we all know, *the more data, the better*.    

## Project goal
The project goal was to reach (around) **0.9** of F1-score. The F1-score below **0.8** was considered as unsatisfactory.

## Results


As the last project in the Data Science program I have chosen the Spark project  
The project data comes in a small sample dataset available on local machine and in a big dataset of 12GB available on Amazon EMR cluster.  


The Sparkify users can stream the music using the free subscription plan with ads or paid subscription plan without ads. Apart from using the service to listen to the music, users can thumb up or down, add songs to playlists, or add friends. Users are free to change their subscription plan by upgrading from free to paid, by downgrading from paid to free, or to entirely stop using the service by cancelling the subscription.



Lastly, we create a simple Flask website that reads the stored model and classifies the messages passed by the end user. Apart from the classification task the application also shows some general visualization graphs of the training dataset and the classifier graph based on the classification result.

## Installation
### Clone
```sh
$ git clone https://github.com/amosvoron/sparkify.git
```

## Libraries
- Spark 2.4.5

## Repository Description

```sh
- app
| - templates
| |- master.html                    # main template of web app
| |- go.html                        # classification result page of web app
|- run.py                           # Flask file that runs app

- data
|- disaster_categories.csv          # input categories data 
|- disaster_messages.csv            # input messages data
|- process_data.py                  # ETL pipeline code
|- DisasterResponse.db              # database to save clean data to

- models
|- train_classifier.py              # ML pipeline code

- notebook
|- ETL Pipeline Preparation.ipynb   # ETL pipeline code (jupyter notebook file)
|- ML Pipeline Preparation.ipynb    # ML pipeline code (jupyter notebook file) 

- Graphs1-3.jpg                     # general graphs
- Graph-4.jpg                       # classsifier graph
- README.md                         # README file
- LICENCE.md                        # LICENCE file
```

## Acknowledgements

## License

MIT