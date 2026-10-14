from sklearn.preprocessing import StandardScaler, LabelEncoder 
df['target'] = df['label'].apply(lambda x: 0 if str(x).strip() == 'normal' else 1)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist() if 'label' in 
categorical_cols: categorical_cols.remove('label') df_encoded = 
pd.get_dummies(df.drop(columns=['label']), columns=categorical_cols)
X = df_encoded.drop(columns=['target']) 
y = df_encoded['target']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Preprocessed Features Shape:", X_scaled.shape) 
print("Normal (0) vs Attack (1) Counts:\n", y.value_counts())



import pandas as pd from sklearn.ensemble import 
RandomForestClassifier from sklearn.feature_selection 
import SelectKBest, chi2
data = {
'duration': [0, 0, 2, 0, 5, 1, 0, 0],
'src_bytes': [181, 239, 235, 219, 217, 217, 0, 0],
'dst_bytes': [5450, 486, 1337, 1337, 2037, 2037, 0, 0],
'wrong_fragment': [0, 0, 0, 0, 0, 0, 3, 1],
'hot': [0, 0, 0, 0, 0, 0, 2, 0],
'label': [0, 0, 0, 0, 0, 0, 1, 1] # 0: Normal, 1: Attack
} df = 
pd.DataFrame(data)
X = df.drop('label', axis=1) 
y = df['label']
selector = SelectKBest(score_func=chi2, k=3) X_new = selector.fit_transform(X, 
y) print("Selected top 3 feature columns:", 
X.columns[selector.get_support()].tolist())
model = RandomForestClassifier(random_state=42) 
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns) 
print("\nFeature Importances:")
print(importances.sort_values(ascending=False))





import pandas as pd from sklearn.model_selection 
import train_test_split from sklearn.tree import 
DecisionTreeClassifier from sklearn.neighbors 
import KNeighborsClassifier from sklearn.metrics 
import accuracy_score
data = {
'duration': [0, 0, 2, 0, 5, 1, 0, 0, 0, 10],
'src_bytes': [181, 239, 235, 219, 217, 217, 0, 0, 0, 5000],
'dst_bytes': [5450, 486, 1337, 1337, 2037, 2037, 0, 0, 0, 0],
'wrong_fragment': [0, 0, 0, 0, 0, 0, 3, 1, 2, 0],
'label': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
} df = 
pd.DataFrame(data)
X = df.drop('label', axis=1) 
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train) 
dt_preds = dt_model.predict(X_test)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train) 
knn_preds = knn_model.predict(X_test) 11
print(f"Decision Tree Accuracy: 
{accuracy_score(y_test, dt_preds) * 
100:.2f}%") print(f"k-NN Accuracy: 
{accuracy_score(y_test, knn_preds) * 
100:.2f}%")





import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, 
confusion_matrix, classification_report
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0] # 1 False Positive, 1 False Negative
acc = accuracy_score(y_true, y_pred) 
prec = precision_score(y_true, y_pred) 
rec = recall_score(y_true, y_pred) f1 = 
f1_score(y_true, y_pred) cm = 
confusion_matrix(y_true, y_pred)
print("Confusion Matrix:") 
print(cm) print(f"\nAccuracy : 
{acc:.2f}") print(f"Precision: 
{prec:.2f}") print(f"Recall : 
{rec:.2f}") print(f"F1-Score : 
{f1:.2f}")
print("\nFull Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))
