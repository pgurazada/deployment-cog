import joblib

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dataset = fetch_openml(data_id=43355,
                       as_frame=True,
                       parser='auto')

diamond_prices = dataset.data

target = 'price'
numeric_features = ['carat']
categorical_features = ['shape', 'cut', 'color', 'clarity', 'report', 'type']

X = diamond_prices.drop(columns=[target, 'id', 'url', 'date_fetched'])
y = diamond_prices[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                test_size=0.2,
                                                random_state=42)

preprocessor = make_column_transformer((StandardScaler(), numeric_features),
                                       (OneHotEncoder(handle_unknown='ignore'), 
                                                      categorical_features))

model_pipeline = make_pipeline(preprocessor, 
                               DecisionTreeRegressor())

if __name__ == "__main__":
    model_pipeline.fit(Xtrain, ytrain)
    saved_model_path = "model-v1.joblib"
    joblib.dump(model_pipeline, saved_model_path)