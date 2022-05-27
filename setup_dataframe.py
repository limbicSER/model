import pandas as pd
from feature_extraction import MFCC


class Setup():
    def __init__(self):
        self.mfcc = MFCC()

    def setup_dataframe(self, gender, features, labels):
        df = pd.DataFrame(features)
        df['labels'] = labels
        df.to_csv(f'csv_files/{gender}_features.csv', index=False)

        print(f'{gender} dataframe')
        df.sample(frac=1).head()

        return df

    def Xysplit(self, data):
        X, y = [], []
        for path, emotion in zip(data.path, data.labels):
            features = self.mfcc.get_features(path)
            # adding augmentation, get_features return a multi dimensional array (for each augmentation), so we have to use a loop to fill the df
            for elem in features:
                X.append(elem)
                y.append(emotion)
        return X, y
