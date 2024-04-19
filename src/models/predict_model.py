import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import pathlib

def load_data(data_path,target,testseed,seed):
    df=pd.read_csv(data_path)
    x=df.drop(columns=target)
    y=df[target]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testseed,random_state=seed)
    return x_train,x_test,y_train,y_test


def model_building(x_train, x_test, y_train, y_test, path):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    try:
        joblib.dump(model, path + "/model.joblib")
    except Exception as e:
        return f"Unable to dump model because of {e}"


def main():
    curr_dir=pathlib.Path(__file__)
    parent_dir=curr_dir.parent.parent.parent
    data_path=parent_dir.as_posix()+'/data/raw/mushroom_cleaned.csv'
    target='class'
    testseed=0.2
    seed=42
    x_train,x_test,y_train,y_test=load_data(data_path,target,testseed,seed)
    path=parent_dir.as_posix()+'/models'
    model_building(x_train,x_test,y_train,y_test,path)
    

if __name__=='__main__':
    main()
