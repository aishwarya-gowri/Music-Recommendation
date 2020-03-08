import pandas
from sklearn.model_selection import train_test_split
import numpy as np
import time
import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation
import pylab as pl
from tkinter import *

triplets_file = "https://static.turi.com/datasets/millionsong/10000.txt"
songs_metadata_file = "https://static.turi.com/datasets/millionsong/song_data.csv"
song_df_1 = pandas.read_table(triplets_file, header=None)
song_df_1.columns = ["user_id", "song_id", "listen_count"]

song_df_2 = pandas.read_csv(songs_metadata_file)
song_df = pandas.merge(
    song_df_1, song_df_2.drop_duplicates(["song_id"]), on="song_id", how="left"
)

# print(song_df.head())

# print(len(song_df))

song_df = song_df.head(10000)

# Merge song title and artist_name columns to make a merged column
song_df["song"] = song_df["title"].map(str) + " - " + song_df["artist_name"]

song_grouped = song_df.groupby(["song"]).agg({"listen_count": "count"}).reset_index()
grouped_sum = song_grouped["listen_count"].sum()
song_grouped["percentage"] = song_grouped["listen_count"].div(grouped_sum) * 100
song_grouped.sort_values(["listen_count", "song"], ascending=[0, 1])

# print(song_grouped)

users = song_df["user_id"].unique()
# print("-------------------sample data--------------------")
train_data, test_data = train_test_split(song_df, test_size=0.20, random_state=0)
# print(train_data.head(5))


####################### Simple popularity-based recommender class  #######################

pm = Recommenders.popularity_recommender_py()
pm.create(train_data, "user_id", "song")


#######################  song recommender with personalization  ##############################


is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, "user_id", "song")


#######################  Quantitative comparison between the models##############################

# print("---------------------Comparing the two recommenders----------------------------")

# start = time.time()

# # Define what percentage of users to use for precision recall calculation
# user_sample = 0.05

# # Instantiate the precision_recall_calculator class
# pr = Evaluation.precision_recall_calculator(test_data, train_data, pm, is_model)

# # Call method to calculate precision and recall values
# (
#     pm_avg_precision_list,
#     pm_avg_recall_list,
#     ism_avg_precision_list,
#     ism_avg_recall_list,
# ) = pr.calculate_measures(user_sample)

# end = time.time()
# print(end - start)

# ####################### Code to plot precision recall curve ##############################

# print(
#     "-----------------------------precision recall curve-------------------------------------"
# )

# # Method to generate precision and recall curve
# def plot_precision_recall(
#     m1_precision_list,
#     m1_recall_list,
#     m1_label,
#     m2_precision_list,
#     m2_recall_list,
#     m2_label,
# ):
#     pl.clf()
#     pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
#     pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
#     pl.xlabel("Recall")
#     pl.ylabel("Precision")
#     pl.ylim([0.0, 0.20])
#     pl.xlim([0.0, 0.20])
#     pl.title("Precision-Recall curve")
#     # pl.legend(loc="upper right")
#     pl.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
#     pl.show()

#     print("Plotting precision recall curves.")


# plot_precision_recall(
#     pm_avg_precision_list,
#     pm_avg_recall_list,
#     "popularity_model",
#     ism_avg_precision_list,
#     ism_avg_recall_list,
#     "item_similarity_model",
# )
choice = 0
while choice != 4:
    print(
        "\n\n\n------------------------------------------------------------WELCOME TO MUSICLY-----------------------------------------------------------------"
    )
    id1 = int(input("Enter the user-id:"))
    user_id = users[id1]
    print(
        "-------------------------------------------------!!!!!!!!!!HELLO!!!!!!!!!! ",
        "-----------------------------------------------------",
    )
    print("\n\nYour Playlist is as follows:\n\n")
    user_items = is_model.get_user_items(user_id)
    for user_item in user_items:
        print(user_item)
    while 1:
        print("\n\nChoose from the below menu")
        choice = int(
            input(
                "\n1.Trending songs\n2.Personalized Recommendation\n3.Logout\n4.Exit\nEnter:"
            )
        )
        if choice == 1:
            print(
                "-------------------popularity-based recommender --------------------"
            )
            # id1 = int(input("Enter the user-id:"))
            # user_id = users[id1]
            # print("Recommending songs for ", id1)
            print(pm.recommend(user_id))

        elif choice == 2:
            print(
                "-------------------song recommender with personalization --------------------\n"
            )
            # id1 = int(input("Enter the user-id:"))
            # user_id = users[id1]
            # print("Recommending songs for ", id1)
            user_items = is_model.get_user_items(user_id)
            # print(
            #     "------------------------------------------------------------------------------------"
            # )
            # print("Training data songs")
            # print(
            #     "------------------------------------------------------------------------------------"
            # )
            # for user_item in user_items:
            #     print(user_item)

            # print("Collabrative recommendation:")
            # print(
            #      "------------------------------------------------------------------------------------"
            #  )
            print(is_model.recommend(user_id))

        elif choice == 3:
            print(
                "------------------------------------------------------------------------------------"
            )
            print("\n\n\t\t\tThank You!!!!\n\t\t\tVisit Again.!!!!\n\n")
            print(
                "------------------------------------------------------------------------------------"
            )
            break

        elif choice == 4:
            exit(0)
        else:
            print("Please choose from the above menu.\nThank you")
