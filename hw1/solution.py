import pandas as pd
df = pd.read_csv('data/home_loans.csv', low_memory=False) # read the csv file into a pandas dataframe object

# Rebalance Database
# Source: https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets)
count_class_0, count_class_1 = df.loan_approved.value_counts()
balanced_count = min(count_class_0, count_class_1)

# Divide by Class
df_class_0 = df[df['loan_approved'] == 0]
df_class_1 = df[df['loan_approved'] == 1]

# Randomly Undersample
df_class_0_under = df_class_0.sample(balanced_count)
df_class_1_under = df_class_1.sample(balanced_count)
df = pd.concat([df_class_0_under, df_class_1_under], axis=0)

import sklearn # import scikit-learn
from sklearn import preprocessing # import preprocessing utilites

features_cat = ['loan_purpose_name', 'applicant_sex_name', 'applicant_ethnicity_name', 'co_applicant_ethnicity_name', 'is_hoepa_loan', 'co_applicant_sex_name', 'property_type_name', 'loan_type_name', 'occupied_by_owner', 'applicant_race_name_1', 'co_applicant_race_name_1']
features_num = ['loan_amount_000s', 'applicant_income_000s']

X_cat = df[features_cat]
X_num = df[features_num]

enc = preprocessing.OneHotEncoder()
enc.fit(X_cat) # fit the encoder to categories in our data 
one_hot = enc.transform(X_cat) # transform data into one hot encoded sparse array format

# Finally, put the newly encoded sparse array back into a pandas dataframe so that we can use it
X_cat_proc = pd.DataFrame(one_hot.toarray(), columns=enc.get_feature_names())
X_cat_proc.head()

scaled = preprocessing.scale(X_num)
X_num_proc = pd.DataFrame(scaled, columns=features_num)
X_num_proc.head()

X = pd.concat([X_num_proc, X_cat_proc], axis=1, sort=False)
X.head()

X = X.fillna(0) # remove NaN values

y = df['loan_approved'] # target

from sklearn.model_selection import train_test_split
X_train, X_TEMP, y_train, y_TEMP = train_test_split(X, y, test_size=0.30) # split out into training 70% of our data
X_validation, X_test, y_validation, y_test = train_test_split(X_TEMP, y_TEMP, test_size=0.50) # split out into validation 15% of our data and test 15% of our data
print(X_train.shape, X_validation.shape, X_test.shape) # print data shape to check the sizing is correct

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# helper method to print basic model metrics
def metrics(y_true, y_pred):
    print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
    print('\nReport:\n', classification_report(y_true, y_pred))


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 8), random_state=1)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_validation)
metrics(y_validation, y_pred)
