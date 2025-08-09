import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Load Iris data and split into train/test sets
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

#Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

#Save the trained model to a file
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved to 'iris_model.pkl'")