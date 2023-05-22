# Using Finetuned Transformers via HuggingFace: Detecting Emotions in Fake and Real News Headlines
Repository link: https://github.com/MinaAlmasi/assignment4-using-finetuned-transformers

This repository forms *assignment 4* by Mina Almasi (202005465) in the subject Language Analytics, Cultural Data Science, F2023. The assignment description can be found [here](https://github.com/MinaAlmasi/assignment4-using-finetuned-transformers/blob/main/assignment-desc.md). 

The repository contains code for running an emotion classification using the BERT model [*j-hartmann/emotion-english-distilroberta-base*](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base). The results are visualized and key differences between the two sets of headlines are discussed in the [Results](https://github.com/MinaAlmasi/assignment4-using-finetuned-transformers#results) section.

## Data
The emotion classifcation is run on ```FAKE``` and ```REAL``` news headlines from the [Fake or Real News](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) dataset. The dataset contains 7796 articles. It is located in the ```data``` folder. 

## Reproducibility
To reproduce the emotion classification and visualisation, follow the instructions in the [Pipeline](https://github.com/MinaAlmasi/assignment4-using-finetuned-transformers#pipeline) section.

## Project Structure
The repository is structured as such:
```
├── README.md
├── assignment-desc.md
├── data                                                
│   ├── README.md  
│   └── fake_or_real_news.csv                           <---     raw data with titles, text, headlines & label
│   └── fake_or_real_news_with_emotion_labels.csv       <---     REAL/FAKE headlines w. emotion labels & scores
├── figures                                             <---     all figures/table in README are stored here
│   ├── emotion_countplot.png       
│   ├── emotion_piecharts.png
│   └── emotion_table.txt
├── requirements.txt 
├── run.sh                                              <---     to reproduce pipeline (classify + visualise)
├── setup.sh                                            <---     creates virtual env, install necessary reqs
└── src
    ├── classify_emotion.py                             <---     perform emotion classification
    └── visualise_emotion.py                            <---     visualise results from classification
```

## Pipeline
The pipeline has been tested on Ubuntu v22.10, Python v3.10.7 ([UCloud](https://cloud.sdu.dk/), Coder Python 1.77.3). Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the pipeline to work.

### Setup
Prior to running the emotion classification, install the necessary packages and requirements in a newly created ```env``` by running ```setup.sh```:
```
bash setup.sh
```

## Running the Classification and Visualisation
To run the full analysis pipeline (including the emotion classification and visualisation), type ```run.sh``` in the terminal:
```
bash run.sh
```


## Results 

### Count Distribution
An overview of the distribution of emotion labels in the news headlines is displayed in the table below: 
|      |   Neutral |   Fear |   Anger |   Sadness |   Disgust |   Surprise |   Joy |
|------|-----------|--------|---------|-----------|-----------|------------|-------|
| ALL  |      3180 |   1076 |     795 |       487 |       434 |        208 |   155 |
| REAL |      1649 |    555 |     383 |       245 |       186 |         90 |    63 |
| FAKE |      1531 |    521 |     412 |       242 |       248 |        118 |    92 |

The distribution is further visualised with two count plots below. The first plot shows the overall distribution (```ALL```) while the second plot groups this distribution by whether the news headline is ```REAL``` or ```FAKE```: 

<p align="left">
  <img src="https://github.com/MinaAlmasi/assignment4-using-finetuned-transformers/blob/main/figures/emotion_countplot.png">
</p>
 

From the first count plot and table, it is evident that a clear majority of the news headlines have been classified as ```Neutral``` by the BERT emotion model with ```Fear``` being the second most classified emotion. 

The second count plot displays no major differences between ```REAL``` and ```FAKE``` news headlines although there are slightly more ```FAKE``` headlines which are classified as the emotions ```Disgust```, ```Anger```, ```Joy``` and ```Surprise```. 

Noteably, the magnitude of the differences between the groups ```REAL``` and ```FAKE``` can be tricky to deduce on a count plot with absolute values when there is an uneven distribution of each group. In this case, the difference in the amount of headlines per group is small, but still present. For this reason, the distribution of emotions are visualised in the next section as a proportion of the total number of headlines in ```REAL``` versus ```FAKE``` news headlines.

### Proportion of Emotions in Real and Fake News Headlines
The pie charts below display the proportion of each emotion in ```REAL``` and ```FAKE``` news headlines, respectively:

<p align="left">
  <img src="https://github.com/MinaAlmasi/assignment4-using-finetuned-transformers/blob/main/figures/emotion_piecharts.png">
</p>

The plots reveal that the proportion of each emotion is very similar between the two groups. 

For ```Neutral```, the difference between ```REAL```  and ```FAKE``` is ```3.6``` percentage points. For the other emotions, the difference is around 1 percentage point in the distribution of emotions between the two groups with ```Disgust``` displaying a slightly higher difference (```1.9``` percentage points difference with the proportion being higher in ```FAKE``` headlines). 

### Discussion and Conclusion
No major differences between ```REAL``` and ```FAKE``` headlines are present in the emotion classification done by [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base). In general, the headlines seem to be predominantly classified as ```Neutral```. 

It should be noted that these classifications are merely the result of one finetuned model, and they do not represent any **ground truth**. We can therefore not conclude that most news headlines are ```Neutral```, but merely that this is the model's "opinion." On the [model card](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) on Hugging Face, it is mentioned that the evaluation accuracy was 66% which is notably higher than the chance level of 14%. Nonetheless, the model is not perfect. 

Two things could be considered for a more nuanced look into these differences. Firstly, it may be of interest to compare the current model with other similar models to investigate whether there is agreement in these classifications. Secondly, each emotion label is given a probability score that is also included in the labelled data (```fake_or_real_news_with_emotion_labels.csv```). It is worth investigating whether there are differences in how certain the model has been in classifying emotions for ```REAL```  versus ```FAKE``` headlines. 

## Author 
This repository was created by Mina Almasi:

* github user: @MinaAlmasi
* student no: 202005465, AUID: au675000
* mail: mina.almasi@post.au.dk
