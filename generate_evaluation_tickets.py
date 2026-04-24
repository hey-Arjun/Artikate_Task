import pandas as pd
import os

train_path = "section_03_classifier/data/training_tickets.csv"
eval_path = "section_03_classifier/data/evaluation_tickets.csv"

if os.path.exists(train_path):
    df = pd.read_csv(train_path)
    # Take 100 rows for evaluation
    eval_df = df.sample(n=100, random_state=42)
    # The rest stays in training
    new_train_df = df.drop(eval_df.index)
    
    eval_df.to_csv(eval_path, index=False)
    new_train_df.to_csv(train_path, index=False)
    print(f" Created {eval_path} with 100 rows and updated training set.")
else:
    print(" training_tickets.csv not found!")