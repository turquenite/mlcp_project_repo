from dataset import MLPC_Dataset
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

dataset = MLPC_Dataset()
X_train, X_test, y_train, y_test = dataset.get_dataset()

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get feature importances
importances = clf.feature_importances_

feature_importances = pd.Series(importances).sort_values(ascending=False)  

# Display the most important features
print("Feature importances:\n", feature_importances)

# Plot the feature importances (optional)
import matplotlib.pyplot as plt
feature_importances.plot(kind='bar', figsize=(12, 6))
plt.title("Feature Importances")
plt.show()