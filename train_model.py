from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle, mlflow, mlflow.sklearn, pandas as pd, os, json, pathlib

# save data to version
iris = load_iris(as_frame=True)
df = pd.concat([iris.data, iris.target], axis=1)
pathlib.Path("data").mkdir(exist_ok=True)
df.to_csv("data/iris.csv", index=False)

# MLflow tracking
mlflow.set_experiment("iris-demo")
with mlflow.start_run():
    X,y = df.iloc[:,:4], df.target
    model = RandomForestClassifier().fit(X,y)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("acc", model.score(X,y))
    pickle.dump(model, open("model.pkl","wb"))
