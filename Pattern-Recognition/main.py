import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mlxtend.preprocessing import minmax_scaling
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

from perceptron import Perceptron
from leastsquares import Least_Squares

# διαβάζω το αρχείο και το αποθηκεύω στη μεταβλητή df
df = pd.read_csv("housing.csv")
# np.set_printoptions(threshold=sys.maxsize)

# γεμίζω τις ελλιπείς τιμές με τη διάμεση τιμή του χαρακτηριστικού
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# χρησιμοποιώ την τεχνική κλιμάκωσης minmax_scaling
# αναπαριστώντας ολα τα χαρακτηριστικά στην ίδια κλίμακα, μεταξύ 0 και 1
scaled_data0 = minmax_scaling(df['longitude'], columns=[0])
scaled_data1 = minmax_scaling(df['latitude'], columns=[0])
scaled_data2 = minmax_scaling(df['housing_median_age'], columns=[0])
scaled_data3 = minmax_scaling(df['total_rooms'], columns=[0])
scaled_data4 = minmax_scaling(df['total_bedrooms'], columns=[0])
scaled_data5 = minmax_scaling(df['population'], columns=[0])
scaled_data6 = minmax_scaling(df['households'], columns=[0])
scaled_data7 = minmax_scaling(df['median_income'], columns=[0])
scaled_data8 = minmax_scaling(df['median_house_value'], columns=[0])

# φτιάχνω τα ιστογράμματα συχνοτήτων για κάθε μια απο τις 10 μεταβλητές

# Πρώτα φτιάχνω τα ιστογράμματα των αριθμητικών χαρακτηριστικών
fig, ax = plt.subplots(9, 2, figsize=(10, 8))
fig.canvas.manager.set_window_title('Numerical features')
# για τη μεταβλητή του γεωγραφικού μήκους (longitude)
sns.histplot(df['longitude'], ax=ax[0][0], kde=True, legend=False)
sns.histplot(scaled_data0, ax=ax[0][1], kde=True, legend=False)
# γεωγραφικού πλάτους (latitude)
sns.histplot(df['latitude'], ax=ax[1][0], kde=True, legend=False)
sns.histplot(scaled_data1, ax=ax[1][1], kde=True, legend=False)
# διάμεση ηλικία των ακινίτων(housing_median_age)
sns.histplot(df['housing_median_age'], ax=ax[2][0], kde=True, legend=False)
sns.histplot(scaled_data2, ax=ax[2][1], kde=True, legend=False)
# συνολικό πλήθος δωματίων (total_rooms)
sns.histplot(df['total_rooms'], ax=ax[3][0], kde=True, legend=False)
sns.histplot(scaled_data3, ax=ax[3][1], kde=True, legend=False)
# συνολικό πλήθος υπνοδωματίων(total_bedrooms)
sns.histplot(df['total_bedrooms'], ax=ax[4][0], kde=True, legend=False)
sns.histplot(scaled_data4, ax=ax[4][1], kde=True, legend=False)
# πληθυσμός (population)
sns.histplot(df['population'], ax=ax[5][0], kde=True, legend=False)
sns.histplot(scaled_data5, ax=ax[5][1], kde=True, legend=False)
# πλήθος νοικοκυριών (households)
sns.histplot(df['households'], ax=ax[6][0], kde=True, legend=False)
sns.histplot(scaled_data6, ax=ax[6][1], kde=True, legend=False)
# διάμεσο εισόδημα κατοίκων (median_income)
sns.histplot(df['median_income'], ax=ax[7][0], kde=True, legend=False)
sns.histplot(scaled_data7, ax=ax[7][1], kde=True, legend=False)

# διάμεση τιμή των ακινήτων (median_house_value)
sns.histplot(df['median_house_value'], ax=ax[8][0], kde=True, legend=False)
sns.histplot(scaled_data8, ax=ax[8][1], kde=True, legend=False)

ax[0][0].set_title("Original Data")
ax[0][1].set_title("Scaled data")
#εμφανίζω τα ιστογράμματα
plt.show()

# integer mapping using LabelEncoder
integer_encoded = LabelEncoder().fit_transform(df['ocean_proximity'])
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

# One hot encoding
onehot_encoded = OneHotEncoder(sparse=False).fit_transform(integer_encoded)

# φτιάχνω τα ιστογράμματα του κατηγορικού χαρακτηριστικού
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.canvas.manager.set_window_title('Categorical feature')
sns.countplot(data=df, x='ocean_proximity', ax=ax[0])
ax[0].set_title('Original Data')
sns.histplot(integer_encoded, ax=ax[1], kde=True, legend=False)
ax[1].set_title('Encoded Data')
plt.show()


# δημιουργώ δισδιάστατα γραφήματα
# με συνδυασμό 2 μεταβλητών
fig, ax = plt.subplots(2, 1, figsize=(10, 5))
fig.canvas.manager.set_window_title('2D graph of 2 variables')
sns.scatterplot(data=df, x='longitude', y='latitude', color='red', edgecolor='black', ax=ax[0])
sns.scatterplot(data=df, x='population', y='median_income', color='purple', edgecolor='black', ax=ax[1])
plt.show()

# scaled data array
features = np.empty((20640, 9))
features = np.concatenate((scaled_data0, scaled_data1, scaled_data2, scaled_data3, scaled_data4, scaled_data5, scaled_data6, scaled_data7,integer_encoded), axis=1)
features_pd = pd.DataFrame(features, columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity'])

# με συνδυασμό 3 μεταβλητών
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.canvas.manager.set_window_title('2D graph of 3 variables')
sns.scatterplot(data=features_pd, x='total_rooms', y='total_bedrooms', size='households', color='blue', edgecolor='black')
plt.show()

# με συνδυασμό 4 μεταβλητών
norm = colors.Normalize(vmin=min(features_pd['ocean_proximity']), vmax=max(features_pd['ocean_proximity']))
cmap = plt.get_cmap('coolwarm')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.canvas.manager.set_window_title('2D graph of 4 variables')
#sns.scatterplot(data=features_pd, x='total_rooms', y='median_income', hue='ocean_proximity', size='population', palette='coolwarm', alpha=0.5)
sns.scatterplot(data=features_pd, x='households', y='median_income', c=features_pd['ocean_proximity'], cmap=cmap, norm=norm , size='housing_median_age', alpha=0.5)
cbar = plt.colorbar(sm)
cbar.set_label('ocean proximity')
plt.show()


# 10-fold cross-validation
kf = KFold(n_splits=10)

X = features
Y = scaled_data8

test_mse = []
test_mae = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    ls = Least_Squares()
    pr = Perceptron()

    # εκπαιδεύω τα μοντέλα χρησιμοποιώντας τα training data
    ls.train(X_train,y_train)
    pr.fit(X_train,y_train)

    # πρόβλεψη της μεταβλητής στόχου για τα test data
    ls_y_pred = ls.predict(X_test)
    pr_y = pr.predict(X_test)

    # υπολογισμός του μεσου τετραγωνικού σφάλματος
    mse_test = np.mean((y_test - ls_y_pred) ** 2)
    # υπολογισμός του μεσου απόλυτου σφάλματος
    mae_test = np.mean(np.absolute(y_test - ls_y_pred))
    test_mse.append(mse_test)
    test_mae.append(mae_test)


avg_test_mse = np.mean(test_mse)
avg_test_mae = np.mean(test_mae)
print("Average test MSE:", avg_test_mse)
print("Average test MAE:", avg_test_mae)
