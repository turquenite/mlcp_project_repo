from dataset import MLPC_Dataset
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils.random import sample_without_replacement

dataset = MLPC_Dataset()
X_train, X_test, y_train, y_test = dataset.get_dataset()

train_size = 10000

subset = sample_without_replacement(len(y_train), train_size)

X_train = X_train[subset]
y_train = y_train[subset]

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
