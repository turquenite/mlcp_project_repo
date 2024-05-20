from dataset import MLPC_Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.random import sample_without_replacement

dataset = MLPC_Dataset()
X_train, X_test, y_train, y_test = dataset.get_dataset()

model = KNeighborsClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')