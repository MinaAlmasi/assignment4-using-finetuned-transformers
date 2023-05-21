'''
Script for Assignment 4, Language Analytics, Cultural Data Science, F2023

Emotion classification on real and fake news headlines using the BERT classifier "j-hartmann/emotion-english-distilroberta-base". 

@MinaAlmasi
'''

# utils
import pathlib
from tqdm import tqdm

# data wrangling 
import pandas as pd 

# huggingface model
from transformers import pipeline

def emotion_classify(classifier, data, text_column:str):
    '''
    Perform emotion classification on text column in a dataframe with an already initalised and fine-tuned huggingface model. 
    Return the emotion with highest probability and its probability score. 

    Args: 
        - classifier: initalised and fine-tuned huggingface model (can be intialised with the transformers' pipeline function)
        - data: the dataframe to classify on
        - text_column: column with text to perform emotion classification on

    Returns: 
        - final_data: dataframe consisting of the original dataframe but with emotion classification ('emotion' column) and probability score ('score' column) added.
    '''

    # empty list to save all emotion predictions
    all_predictions = []

    # iterate over each text in the dataframe's text_column, extract prediction for each text
    for text in tqdm(data[text_column], desc="Performing classification"):
        # create prediction 
        prediction = classifier(text)
        
        # append prediction to list 
        all_predictions.append(prediction[0][0]) # to access first prediction as its dictionary (as the default two list nested e.g. [[{'label': 'fear', 'score': 0.9332622289657593}]])

    # make predictions dataframe 
    predictions_data = pd.DataFrame(all_predictions)

    # rename label into "emotion" to avoid clashing with other labels in data
    predictions_data = predictions_data.rename(columns={"label": "emotion"})

    # combine predictions_data with text and label col in original dataframe
    final_data = pd.concat([data, predictions_data], axis=1)

    return final_data

def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[1] / "data"

    # load data (only loading the relevant columns)
    data = pd.read_csv(datapath / "fake_or_real_news.csv", usecols=["title", "label"])

    # initialize pipeline
    classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores=True,
                      top_k = 1 
                      )

    # do emotion classification
    emotion_data = emotion_classify(classifier, data, "title")

    # save data
    emotion_data.to_csv(datapath / "fake_or_real_news_with_emotion_labels.csv")

    # print statement
    print("[INFO:] Emotion classification completed!")

if __name__ == "__main__":
    main()