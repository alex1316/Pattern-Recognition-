import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Διαβάζω το αρχείο και το αποθηκεύω στη μεταβλητή data
data = pd.read_csv("housing.csv")

# Διαχωρίζω τα δεδομένα σε χαρακτηριστικά και στόχο
features = data.drop("median_house_value", axis=1)
target = data["median_house_value"]

# Διαχωρίζω τα δεδομένα σε δεδομένα εκπαίδευσης και ελέγχου
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Ορίζω την επεξεργασία που θα ακολουθήσει για τα αριθμητικά και τα κατηγορικά χαρακτηριστικά
numeric_transformer = SimpleImputer(strategy="mean")
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder())])

# Ορίζω σε ποιες στήλες βρίσκονται
preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]),
        ("cat", categorical_transformer, ["ocean_proximity"])])

# Δημιουργώ το τελικό pipeline και ορίζω το πολυστρωματικό νευρωνικό δίκτυο
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                          ("regressor", MLPRegressor(hidden_layer_sizes=(50,100,50), max_iter=500))])

# Εκπαιδεύω το pipeline με τα δεδομένα εκπαίδευσης
pipeline.fit(X_train, y_train)

# και αξιολογώ τελικά το μοντέλο με τα δεδομένα ελέγχου
test_score = pipeline.score(X_test, y_test)
print("Mean training score: ", test_score)
