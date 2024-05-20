from dataset import MLPC_Dataset
from sklearn import svm
from sklearn.metrics import accuracy_score

dataset = MLPC_Dataset()
X_train, X_test, y_train, y_test = dataset.get_dataset()

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
