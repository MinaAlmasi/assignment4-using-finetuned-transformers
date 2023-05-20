'''

Script for Assignment 4, Language Analytics, Cultural Data Science, F2023

Visualise emotion labels in fake and real news headlines given by the BERT classifier (j-hartmann/emotion-english-distilroberta-base) used in src/classify_emotion.py.

The script creates the following: 
1. A table of emotions per fake and real news headlines in a README friendly format saved as .txt file.
2. A countplot of all emotions and a countplot of emotions per fake and real news saved as one .png file.
3. Two piecharts showing the proportion of emotions, one for each label saved as one .png file.

@MinaAlmasi
'''

# utils
import pathlib

# data wrangling 
import pandas as pd 

# for creating tables with format to be inserted in README (https://pypi.org/project/tabulate/)
from tabulate import tabulate 

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

def create_emotion_table(data, savepath:pathlib.Path):
    '''
    Create table of emotions per fake and real news in a README friendly format. 
    Write table to .txt file. 

    Args: 
        - data: dataframe with emotion labels
        - savepath: path to write table to

    Output:
        - .txt file with table
    '''

    # make table of emotions per fake and real 
    count_table = pd.crosstab(index=data['emotion'],
                             columns=data['label'], 
                             )

    # make an ALL column for total count of each emotion
    count_table["ALL"] = count_table.sum(axis=1)

    # sort table by ALL column
    count_table = count_table.sort_values(by="ALL", ascending=False)

    # create labels
    emotion_labels = [label.title() for label in count_table.index] # for emotion label in the count_table, capitalize that label using title()

    # create table
    table = tabulate(
        [["ALL"] + count_table["ALL"].tolist(),
        ["REAL"] + count_table["REAL"].tolist(), 
        ["FAKE"] + count_table["FAKE"].tolist()], 
        headers=emotion_labels, 
        tablefmt="github"
    )

    # write table to txt 
    with open(savepath, 'w') as file:
        file.write(table)

def create_emotion_countplot(data, savepath:pathlib.Path):
    '''
    Creates two countplots of emotions. One across all news articles and one across fake and real news.
    Save figure to .png file.
    
    Args:
        - data: dataframe with emotion labels
        - savepath: path to write figure to

    Output:
        - .png file with figure
    '''

    # count amount of news articles per emotion, sort emotions by count to create a order for the countplots rising from high to low
    data["emotion"] = data["emotion"].str.title() # capitalize labels

    order = data.groupby('emotion').count().sort_values(by='label', ascending=False).reset_index() 
    order = order['emotion'].tolist() # convert order to list

    # define color palette
    colors = sns.color_palette("hls", 8)

    # create two subplots with overall figsize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # adjust whitespace, so subplots are not plotted on top of overall figure title
    fig.subplots_adjust(top=0.85)

    # plot all news
    sns.countplot(ax=axes[0], x='emotion', data=data, order=order, palette=colors, width=0.6)
    axes[0].set_title("All News Headlines", fontsize=15)

    # define hue_order for second subplot (so real news are plotted first)
    hue_order = ["REAL", "FAKE"]

    # plot fake vs real news
    sns.countplot(ax=axes[1], x='emotion', data=data, order=order, hue_order=hue_order, palette=["#DB5F57", "#EDAFAB"], hue='label', width=0.6)
    axes[1].set_title("Real versus Fake Headlines", fontsize=15)

    # set labels for both subplots (for loop iterating over ax in both subplots as the labels are identical in both)
    for ax in axes:
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")

    # set legend for second subplot
    axes[1].legend(title="Label", loc='upper right')

    # set overall title with padding
    fig.suptitle("Emotion Labels in Real and Fake News Headlines", fontsize=18, fontweight="bold")

    # save figure
    fig.savefig(savepath, dpi=300)


def create_emotion_piecharts(data, savepath:pathlib.Path):
    '''
    Creates two piecharts of emotions. One across all news articles and one across fake and real news.

    Args:
        - data: dataframe with emotion labels
        - savepath: path to write figure to

    Output:
        - .png file with figure
    '''

    # make table of emotions per fake and real 
    count_table = pd.crosstab(index=data['emotion'],
                             columns=data['label'])

    # create labels
    emotion_labels = [label.title() for label in count_table.index]

    # define color palette (manually to match colors in countplot)
    colors = ["#91db57","#57d3db", "#dbc257", "#a157db", "#db5f57", "#57db80", "#5770db"]

    # create two subplots with overall figsize
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # create pie charts by iterating over the two labels and the two axes:
    for ax, label in enumerate(["REAL", "FAKE"]):
        # make pie chart
        axes[ax].pie(count_table[label], labels=emotion_labels, 
            autopct='%1.1f%%', startangle=90, colors=colors, 
            pctdistance=0.85, textprops={'fontsize': 10, 'fontweight': 'bold'}, 
            wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
            )
        
        # equal aspect ratio for nicer layout
        axes[ax].axis('equal')  

        # add title to each piechart 
        axes[ax].set_title(f"{label.title()} Headlines", fontsize=20, pad=15)

        # change position of piechart to make room for overall
        original_pos = axes[ax].get_position() # get current position for subplot
        new_pos = [original_pos.x0, (original_pos.y0 -0.065),  original_pos.width, original_pos.height] # move subplot down by 0.1

        # move each piechart down to make room for overall
        axes[ax].set_position(new_pos)

    # set overall title with padding
    fig.suptitle("Proportion of Emotion Labels in Real and Fake News Headlines", fontsize=25, fontweight="bold")

    # save figure
    fig.savefig(savepath, dpi=300)


def main():
    # define paths 
    path = pathlib.Path(__file__)
    datapath = path.parents[1] / "data" / "fake_or_real_news_with_emotion_labels.csv"
    outpath = path.parents[1] / "figures"

    # read in data
    data = pd.read_csv(datapath, index_col=[0]) # index_col=[0] to avoid "Unammed:0" column

    # create table
    create_emotion_table(data, outpath / "emotion_table.txt")

    # create countplot of emotions across fake and real news, 
    create_emotion_countplot(data, outpath / "emotion_countplot.png")

    # create piecharts of emotions across fake and real news,
    create_emotion_piecharts(data, outpath / "emotion_piecharts.png")

# run script
if __name__ == "__main__":
    main()