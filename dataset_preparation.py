import os
import pandas as pd


class Datatset:
    def __init__(self):
        self.BASE = "C:/Users/ajida/Desktop/projects/DIPLOMA/ravdess"
        self.RAV = f"{self.BASE}/audio_speech_actors_01-24/"
        self.dir = os.listdir(self.RAV)
        self.males = []
        self.females = []

    def create_dfs(self):
        for actor in self.dir:
            files = os.listdir(self.RAV + actor)
            for file in files:
                part = file.split('.')[0]
                part = part.split("-")

                temp = int(part[6])

                if part[2] == '01':
                    emotion = 'neutral'
                elif part[2] == '02':
                    emotion = 'calm'
                elif part[2] == '03':
                    emotion = 'happy'
                elif part[2] == '04':
                    emotion = 'sad'
                elif part[2] == '05':
                    emotion = 'angry'
                elif part[2] == '06':
                    emotion = 'fear'
                elif part[2] == '07':
                    emotion = 'disgust'
                elif part[2] == '08':
                    emotion = 'surprise'
                else:
                    emotion = 'unknown'

                if temp % 2 == 0:
                    path = (self.RAV + actor + '/' + file)
                    # emotion = 'female_'+emotion
                    self.females.append([emotion, path])
                else:
                    path = (self.RAV + actor + '/' + file)
                    # emotion = 'male_'+emotion
                    self.males.append([emotion, path])

        rav_females_df = pd.DataFrame(self.females)
        rav_females_df.columns = ['labels', 'path']

        rav_males_df = pd.DataFrame(self.males)
        rav_males_df.columns = ['labels', 'path']

        rav_males_df.to_csv("csv_files/males_emotions_df.csv", index=False)
        rav_females_df.to_csv("csv_files/females_emotions_df.csv", index=False)

        return rav_females_df, rav_males_df
