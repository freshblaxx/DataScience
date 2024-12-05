import pandas as pd

# Originaldateien laden
test_df = pd.read_csv("/Users/dominikfrank/Desktop/University/Master/Semester 1/PII/Data Science/Code for Project/DataScience/Projeto/Modeling/Arrests_testing_data_arrest_decision_tree.csv")
train_df = pd.read_csv("/Users/dominikfrank/Desktop/University/Master/Semester 1/PII/Data Science/Code for Project/DataScience/Projeto/Modeling/Arrests_training_data_arrest_decision_tree.csv")

# Ziel: 7000 Trainingsdaten und 3000 Testdaten
train_sample = train_df.sample(n=7000, random_state=42)  # Exakte Anzahl der Trainingsdaten
test_sample = test_df.sample(n=3000, random_state=42)    # Exakte Anzahl der Testdaten

# Gesampelte Dateien speichern
train_sample.to_csv("/Users/dominikfrank/Desktop/University/Master/Semester 1/PII/Data Science/Code for Project/DataScience/Projeto/Sampling/sampled_train_data.csv", index=False)
test_sample.to_csv("/Users/dominikfrank/Desktop/University/Master/Semester 1/PII/Data Science/Code for Project/DataScience/Projeto/Sampling/sampled_test_data.csv", index=False)

# Informationen ausgeben
print(f"Original Trainingsdaten: {len(train_df)}, Gesampelte Trainingsdaten: {len(train_sample)}")
print(f"Original Testdaten: {len(test_df)}, Gesampelte Testdaten: {len(test_sample)}")
print("Gesampelte Dateien wurden gespeichert als:")
print("/Users/dominikfrank/Desktop/University/Master/Semester 1/PII/Data Science/Code for Project/DataScience/Projeto/Sampling/sampled_train_data.csv")
print("/Users/dominikfrank/Desktop/University/Master/Semester 1/PII/Data Science/Code for Project/DataScience/Projeto/Sampling/sampled_test_data.csv")
