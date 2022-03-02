import joblib
import numpy as np

clf=joblib.load('saved_models/tfidf_rf/tfidf_rf_4000_clf.joblib')

importances = clf.named_steps["clf"].feature_importances_
importances = np.argsort(importances)[::-1]
feature_names = clf.named_steps["vect"].get_feature_names_out()  
# print(feature_names)
top_words = []

for i in range(20):
    top_words.append(feature_names[importances[i]])
print(top_words)