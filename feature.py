import pandas as pd
import os

cp  = pd.read_csv('Datasets/content_polluters.txt', sep='\t',header=None)
lu  = pd.read_csv('Datasets/legitimate_users.txt', sep='\t',header=None)
cpt = pd.read_csv('Datasets/content_polluters_tweets.txt', sep='\t',header=None)
cpf = pd.read_csv('Datasets/content_polluters_followings.txt', sep='\t', header=None)
luf = pd.read_csv('Datasets/legitimate_users_followings.txt', sep='\t', header=None)
lut = pd.read_csv('Datasets/legitimate_users_tweets.txt', sep='\t', header=None)

polluters_followings = pd.DataFrame({
    'UserId' : cp.iloc[:, 0],

    #-1- Longueur du nom d’utilisateur 
    'LengthOfScreenName' : cp.iloc[:, 6],

    #-2- Longueur de la description du profil 
    'LengthOfDescriptionInUserProfile' : cp.iloc[:,7],

    #-3- Durée de vie du compte
    'AccountLongevity': ((pd.to_datetime(cp.iloc[:,2]) - pd.to_datetime(cp.iloc[:,1])).dt.days),

    #-4- Nombre de following 
    'NumerOfFollowings' : cp.iloc[:, 3],

    #-5- Nombre de followers
    'NumberOfFollowers' : cp.iloc[:, 4],

    #-6- Rapport « following/followers »
    'RatioFollowingFollowers' : cp.iloc[:, 3] / (cp.iloc[:, 4] + 1) 
})

legitimate_followings = pd.DataFrame({
    'UserId' : lu.iloc[:, 0],

    #-1- Longueur du nom d’utilisateur 
    'LengthOfScreenName' : lu.iloc[:, 6],

    #-2- Longueur de la description du profil
    'LengthOfDescriptionInUserProfile' : lu.iloc[:,7],

    #-3- Durée de vie du compte
    'AccountLongevity': ((pd.to_datetime(lu.iloc[:,2]) - pd.to_datetime(lu.iloc[:,1])).dt.days),

    #-4- Nombre de following
    'NumerOfFollowings' : lu.iloc[:, 3],

    #-5- Nombre de followers
    'NumberOfFollowers' : lu.iloc[:, 4],

    #-6- Rapport « following/followers »
    'RatioFollowingFollowers' : lu.iloc[:, 3] / (lu.iloc[:, 4]+1)
})

#-7- Nombre moyen de tweets par jour
polluters_followings["TweetsPerDay"] = polluters_followings["NumerOfFollowings"] / (polluters_followings["AccountLongevity"] + 1)
legitimate_followings["TweetsPerDay"] = legitimate_followings["NumerOfFollowings"] / (legitimate_followings["AccountLongevity"] + 1)

#-8-Proportion d’URL dans les tweets
#Détection des URLs dans les tweets
cpt["HasURL"] = cpt.iloc[:, 2].str.contains(r'http://|https://', regex=True, na=False)
lut["HasURL"] = lut.iloc[:, 2].str.contains(r'http://|https://', regex=True, na=False)
#Nombre de tweets contenant une URL par utilisateur
polluters_url_counts = cpt.groupby(0)["HasURL"].sum().reset_index()
legitimate_url_counts = lut.groupby(0)["HasURL"].sum().reset_index()
#Renommage des colonnes pour correspondre aux UserID
polluters_url_counts.columns = ["UserId", "URLCount"]
legitimate_url_counts.columns = ["UserId", "URLCount"]
#Fusion avec les DataFrames existants
polluters_followings = polluters_followings.merge(polluters_url_counts, on="UserId", how="left").fillna(0)
legitimate_followings = legitimate_followings.merge(legitimate_url_counts, on="UserId", how="left").fillna(0)
#Calcul du rapport URL/Tweets
polluters_followings["URLRatio"] = polluters_followings["URLCount"] / (cp.iloc[:, 5] + 1)
legitimate_followings["URLRatio"] = legitimate_followings["URLCount"] / (lu.iloc[:, 5] + 1)
#Suppression de la colonne intermédiaire `URLCount` (elle ne sera pas enregistrée)
polluters_followings.drop(columns=["URLCount"], inplace=True)
legitimate_followings.drop(columns=["URLCount"], inplace=True)


# -9- Proportion de mentions @ dans les tweets
# Fonction pour compter le nombre de mentions dans un tweet
def count_mentions(tweet):
    if isinstance(tweet, str):
        words = tweet.split()
        mentions = sum(1 for word in words if word.startswith('@'))
        return mentions
    return 0
# Calcul du nombre total de mentions par utilisateur
cpt["MentionCount"] = cpt.iloc[:, 2].apply(count_mentions)
lut["MentionCount"] = lut.iloc[:, 2].apply(count_mentions)
polluters_mentions_counts = cpt.groupby(0)["MentionCount"].sum().reset_index()
legitimate_mentions_counts = lut.groupby(0)["MentionCount"].sum().reset_index()
# Renommage des colonnes
polluters_mentions_counts.columns = ["UserId", "MentionCount"]
legitimate_mentions_counts.columns = ["UserId", "MentionCount"]
# Fusion avec le DataFrame principal
polluters_followings = polluters_followings.merge(polluters_mentions_counts, on="UserId", how="left").fillna(0)
legitimate_followings = legitimate_followings.merge(legitimate_mentions_counts, on="UserId", how="left").fillna(0)
# Calcul du rapport Mentions/Tweets
polluters_followings["MentionRatio"] = polluters_followings["MentionCount"] / (cp.iloc[:, 5] )
legitimate_followings["MentionRatio"] = legitimate_followings["MentionCount"] / (lu.iloc[:, 5] )
# Suppression de la colonne intermédiaire `MentionCount`
polluters_followings.drop(columns=["MentionCount"], inplace=True)
legitimate_followings.drop(columns=["MentionCount"], inplace=True)


##-10-11-Temps moyen et maximal entre deux tweets consécutifs
# Nombre moyen de tweets par jour
polluters_followings["TweetsPerDay"] = polluters_followings["NumerOfFollowings"] / (polluters_followings["AccountLongevity"] + 1)
legitimate_followings["TweetsPerDay"] = legitimate_followings["NumerOfFollowings"] / (legitimate_followings["AccountLongevity"] + 1)
# Conversion de la colonne de timestamps en datetime
cpt["CreatedAt"] = pd.to_datetime(cpt.iloc[:, 3])
lut["CreatedAt"] = pd.to_datetime(lut.iloc[:, 3])
# Tri des tweets par UserId et date
cpt = cpt.sort_values(by=[0, "CreatedAt"])
lut = lut.sort_values(by=[0, "CreatedAt"])
# Calcul des intervalles de temps entre tweets successifs
cpt["TimeDiff"] = cpt.groupby(0)["CreatedAt"].diff().dt.total_seconds() / 60  # en minutes
lut["TimeDiff"] = lut.groupby(0)["CreatedAt"].diff().dt.total_seconds() / 60  # en minutes
# Remplacer NaN par 0 (premier tweet n'a pas d'intervalle)
#cpt["TimeDiff"].fillna(0.1)
#lut["TimeDiff"].fillna(0.1)
# Calcul du temps moyen et maximal entre tweets pour chaque utilisateur
polluters_time_stats = cpt.groupby(0)["TimeDiff"].agg(["mean", "max"]).reset_index()
legitimate_time_stats = lut.groupby(0)["TimeDiff"].agg(["mean", "max"]).reset_index()
# Renommage des colonnes
polluters_time_stats.columns = ["UserId", "MeanTimeBetweenTweets", "MaxTimeBetweenTweets"]
legitimate_time_stats.columns = ["UserId", "MeanTimeBetweenTweets", "MaxTimeBetweenTweets"]
# Fusion avec les DataFrames principaux
polluters_followings = polluters_followings.merge(polluters_time_stats, on="UserId", how="left").fillna(0)
legitimate_followings = legitimate_followings.merge(legitimate_time_stats, on="UserId", how="left").fillna(0)


# -1️2- Ajout de la caractéristique : Taux d'utilisation des hashtags (HashtagRatio)**
def count_hashtags(tweet):
    if isinstance(tweet, str):
        return tweet.count("#")
    return 0
cpt["HashtagCount"] = cpt.iloc[:, 2].apply(count_hashtags)
lut["HashtagCount"] = lut.iloc[:, 2].apply(count_hashtags)
polluters_hashtag_counts = cpt.groupby(0)["HashtagCount"].sum().reset_index()
legitimate_hashtag_counts = lut.groupby(0)["HashtagCount"].sum().reset_index()
polluters_hashtag_counts.columns = ["UserId", "HashtagCount"]
legitimate_hashtag_counts.columns = ["UserId", "HashtagCount"]
polluters_followings = polluters_followings.merge(polluters_hashtag_counts, on="UserId", how="left").fillna(0)
legitimate_followings = legitimate_followings.merge(legitimate_hashtag_counts, on="UserId", how="left").fillna(0)
polluters_followings["HashtagRatio"] = polluters_followings["HashtagCount"] / (cp.iloc[:, 5] + 1)
legitimate_followings["HashtagRatio"] = legitimate_followings["HashtagCount"] / (lu.iloc[:, 5] + 1)
polluters_followings.drop(columns=["HashtagCount"], inplace=True)
legitimate_followings.drop(columns=["HashtagCount"], inplace=True)
# **-13- Ajout de la caractéristique : Ratio Follow-back (FollowBackRatio)**
polluters_followings["FollowBackRatio"] = polluters_followings["NumberOfFollowers"] / (polluters_followings["NumerOfFollowings"] + 1)
legitimate_followings["FollowBackRatio"] = legitimate_followings["NumberOfFollowers"] / (legitimate_followings["NumerOfFollowings"] + 1)

# Définition du chemin du dossier de sortie
output_dir = "Datatest/Tache2/Partie1"

# Vérification et création du dossier si nécessaire
os.makedirs(output_dir, exist_ok=True)

# Définition des noms des fichiers de sortie
polluters_filename = os.path.join(output_dir, "polluters_features.csv")
legitimate_filename = os.path.join(output_dir, "legitimate_features.csv")

# Enregistrement des DataFrames avec les noms des colonnes
polluters_followings.to_csv(polluters_filename, index=False, sep=',', encoding='utf-8')
legitimate_followings.to_csv(legitimate_filename, index=False, sep=',', encoding='utf-8')

print(f"Fichiers enregistrés :\n - {polluters_filename}\n - {legitimate_filename}")



print("-----------------")
print(polluters_followings.head(10))
print("-----------------")
print(legitimate_followings.head(10))
