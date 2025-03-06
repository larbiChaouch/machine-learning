import pandas as pd

class Preprocessing:
    def __init__(self, polluters_file, legitimate_file, polluters_tweets, legitimate_tweets):
        self.cp = pd.read_csv(polluters_file, sep='\t', header=None)
        self.lu = pd.read_csv(legitimate_file, sep='\t', header=None)
        self.cpt = pd.read_csv(polluters_tweets, sep='\t', header=None)
        self.lut = pd.read_csv(legitimate_tweets, sep='\t', header=None)
        self.polluters_df = pd.DataFrame()
        self.legitimate_df = pd.DataFrame()
    
    def extract_basic_features(self):
        self.polluters_df['UserId'] = self.cp.iloc[:, 0]
        self.legitimate_df['UserId'] = self.lu.iloc[:, 0]
        self.polluters_df['LengthOfScreenName'] = self.cp.iloc[:, 6]
        self.legitimate_df['LengthOfScreenName'] = self.lu.iloc[:, 6]
        self.polluters_df['LengthOfDescriptionInUserProfile'] = self.cp.iloc[:, 7]
        self.legitimate_df['LengthOfDescriptionInUserProfile'] = self.lu.iloc[:, 7]
        self.polluters_df['AccountLongevity'] = (pd.to_datetime(self.cp.iloc[:, 2]) - pd.to_datetime(self.cp.iloc[:, 1])).dt.days
        self.legitimate_df['AccountLongevity'] = (pd.to_datetime(self.lu.iloc[:, 2]) - pd.to_datetime(self.lu.iloc[:, 1])).dt.days
        self.polluters_df['NumerOfFollowings'] = self.cp.iloc[:, 3]
        self.legitimate_df['NumerOfFollowings'] = self.lu.iloc[:, 3]
        self.polluters_df['NumberOfFollowers'] = self.cp.iloc[:, 4]
        self.legitimate_df['NumberOfFollowers'] = self.lu.iloc[:, 4]
        self.polluters_df['RatioFollowingFollowers'] = self.polluters_df['NumerOfFollowings'] / (self.polluters_df['NumberOfFollowers'] + 1)
        self.legitimate_df['RatioFollowingFollowers'] = self.legitimate_df['NumerOfFollowings'] / (self.legitimate_df['NumberOfFollowers'] + 1)
    
    def extract_tweets_per_day(self):
        self.polluters_df['TweetsPerDay'] = self.polluters_df['NumerOfFollowings'] / (self.polluters_df['AccountLongevity'] + 1)
        self.legitimate_df['TweetsPerDay'] = self.legitimate_df['NumerOfFollowings'] / (self.legitimate_df['AccountLongevity'] + 1)
    
    '''def extract_url_ratio(self):
        """Extraction du ratio d'URL dans les tweets"""
        for df, tweets, source in [(self.polluters_df, self.cpt, self.cp), (self.legitimate_df, self.lut, self.lu)]:
            # Vérification de la présence de UserId avant la fusion
            if 'UserId' not in df.columns:
                print("⚠️ Erreur : UserId absent du DataFrame cible avant fusion")
                print(df.head())

            # Détection des URLs dans les tweets
            tweets['HasURL'] = tweets.iloc[:, 2].str.contains(r'http://|https://', regex=True, na=False)

            # Nombre de tweets contenant une URL par utilisateur
            url_counts = tweets.groupby(0)['HasURL'].sum().reset_index()
            url_counts.columns = ['UserId', 'URLCount']

            # Vérification avant fusion
            print("✅ Vérification avant fusion :")
            print(url_counts.head())

            # Fusion avec le DataFrame existant
            df = df.merge(url_counts, on='UserId', how='left')

            # Vérification après fusion
            if 'URLCount' not in df.columns:
                print("❌ ERREUR : La fusion a échoué, URLCount absent")
            else:
                print("✅ Fusion réussie, URLCount ajouté")

            # Remplissage des valeurs manquantes
            df.fillna({'URLCount': 0}, inplace=True)

            # Calcul du ratio d'URL
            df['URLRatio'] = df['URLCount'] / (source.iloc[:, 5] + 1)

            # Suppression de la colonne intermédiaire
            df.drop(columns=['URLCount'], inplace=True)

            # Sauvegarde dans l'objet
            if 'polluters_df' in locals():
                self.polluters_df = df
            else:
                self.legitimate_df = df'''
    
    def extract_url_ratio(self):
        """Extraction du ratio d'URL dans les tweets"""
        for data_type in ['polluters', 'legitimate']:
            if data_type == 'polluters':
                df, tweets, source = self.polluters_df, self.cpt, self.cp
            else:
                df, tweets, source = self.legitimate_df, self.lut, self.lu

            tweets['HasURL'] = tweets.iloc[:, 2].str.contains(r'http://|https://', regex=True, na=False)
            url_counts = tweets.groupby(0)['HasURL'].sum().reset_index()
            url_counts.columns = ['UserId', 'URLCount']

            df = df.merge(url_counts, on='UserId', how='left').fillna(0)
            df['URLRatio'] = df['URLCount'] / (source.iloc[:, 5] + 1)
            df.drop(columns=['URLCount'], inplace=True)

            # ✅ Mise à jour des DataFrames
            if data_type == 'polluters':
                self.polluters_df = df
            else:
                self.legitimate_df = df


    def extract_hashtag_ratio(self):
        """Extraction du ratio d'utilisation des hashtags"""
        def count_hashtags(tweet):
            return tweet.count("#") if isinstance(tweet, str) else 0

        self.cpt["HashtagCount"] = self.cpt.iloc[:, 2].apply(count_hashtags)
        self.lut["HashtagCount"] = self.lut.iloc[:, 2].apply(count_hashtags)

        polluters_hashtag_counts = self.cpt.groupby(0)["HashtagCount"].sum().reset_index()
        legitimate_hashtag_counts = self.lut.groupby(0)["HashtagCount"].sum().reset_index()

        polluters_hashtag_counts.columns = ["UserId", "HashtagCount"]
        legitimate_hashtag_counts.columns = ["UserId", "HashtagCount"]

        self.polluters_df = self.polluters_df.merge(polluters_hashtag_counts, on="UserId", how="left").fillna(0)
        self.legitimate_df = self.legitimate_df.merge(legitimate_hashtag_counts, on="UserId", how="left").fillna(0)

        self.polluters_df["HashtagRatio"] = self.polluters_df["HashtagCount"] / (self.cp.iloc[:, 5] + 1)
        self.legitimate_df["HashtagRatio"] = self.legitimate_df["HashtagCount"] / (self.lu.iloc[:, 5] + 1)

        self.polluters_df.drop(columns=["HashtagCount"], inplace=True)
        self.legitimate_df.drop(columns=["HashtagCount"], inplace=True)


    def extract_follow_back_ratio(self):
        """Ajout du ratio FollowBack (Nombre de followers / Nombre de followings)"""
        self.polluters_df["FollowBackRatio"] = self.polluters_df["NumberOfFollowers"] / (self.polluters_df["NumerOfFollowings"] + 1)
        self.legitimate_df["FollowBackRatio"] = self.legitimate_df["NumberOfFollowers"] / (self.legitimate_df["NumerOfFollowings"] + 1)

    
    def extract_mentions_ratio(self):
        def count_mentions(tweet):
            return sum(1 for word in tweet.split() if word.startswith('@')) if isinstance(tweet, str) else 0
        self.cpt['MentionCount'] = self.cpt.iloc[:, 2].apply(count_mentions)
        self.lut['MentionCount'] = self.lut.iloc[:, 2].apply(count_mentions)
        polluters_mentions_counts = self.cpt.groupby(0)['MentionCount'].sum().reset_index()
        legitimate_mentions_counts = self.lut.groupby(0)['MentionCount'].sum().reset_index()
        polluters_mentions_counts.columns = ['UserId', 'MentionCount']
        legitimate_mentions_counts.columns = ['UserId', 'MentionCount']
        self.polluters_df = self.polluters_df.merge(polluters_mentions_counts, on='UserId', how='left').fillna(0)
        self.legitimate_df = self.legitimate_df.merge(legitimate_mentions_counts, on='UserId', how='left').fillna(0)
        self.polluters_df['MentionRatio'] = self.polluters_df['MentionCount'] / self.cp.iloc[:, 5]
        self.legitimate_df['MentionRatio'] = self.legitimate_df['MentionCount'] / self.lu.iloc[:, 5]
        self.polluters_df.drop(columns=['MentionCount'], inplace=True)
        self.legitimate_df.drop(columns=['MentionCount'], inplace=True)
    
    def extract_time_between_tweets(self):
        self.cpt['CreatedAt'] = pd.to_datetime(self.cpt.iloc[:, 3])
        self.lut['CreatedAt'] = pd.to_datetime(self.lut.iloc[:, 3])
        self.cpt['TimeDiff'] = self.cpt.groupby(0)['CreatedAt'].diff().dt.total_seconds().abs() / 60
        self.lut['TimeDiff'] = self.lut.groupby(0)['CreatedAt'].diff().dt.total_seconds().abs() / 60
        self.cpt['TimeDiff'].fillna(0)
        self.lut['TimeDiff'].fillna(0)
        polluters_time_stats = self.cpt.groupby(0)['TimeDiff'].agg(['mean', 'max']).reset_index()
        legitimate_time_stats = self.lut.groupby(0)['TimeDiff'].agg(['mean', 'max']).reset_index()
        polluters_time_stats.columns = ['UserId', 'MeanTimeBetweenTweets', 'MaxTimeBetweenTweets']
        legitimate_time_stats.columns = ['UserId', 'MeanTimeBetweenTweets', 'MaxTimeBetweenTweets']
        self.polluters_df = self.polluters_df.merge(polluters_time_stats, on='UserId', how='left').fillna(0)
        self.legitimate_df = self.legitimate_df.merge(legitimate_time_stats, on='UserId', how='left').fillna(0)
    
    def run_all_extractions(self):
        self.extract_basic_features()
        self.extract_tweets_per_day()
        self.extract_url_ratio()
        self.extract_mentions_ratio()
        self.extract_time_between_tweets()
        self.extract_hashtag_ratio()  
        self.extract_follow_back_ratio()  
    
    def display_results(self):
        print("-----------------")
        print(self.polluters_df.head(10))
        print("-----------------")
        print(self.legitimate_df.head(10))

if __name__ == "__main__":
    extractor = Preprocessing(
        'Datasets/content_polluters.txt', 'Datasets/legitimate_users.txt',
        'Datasets/content_polluters_tweets.txt', 'Datasets/legitimate_users_tweets.txt'
        )
    extractor.run_all_extractions()
    extractor.display_results()
