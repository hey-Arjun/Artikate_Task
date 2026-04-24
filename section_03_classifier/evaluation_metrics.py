import pandas as pd
from classifier import TicketClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def run_evaluation():
    eval_df = pd.read_csv("section_03_classifier/data/evaluation_tickets.csv")
    classifier = TicketClassifier()
    
    print("Predicting categories for evaluation set...")
    eval_df['prediction'] = eval_df['text'].apply(classifier.predict)
    
    report = classification_report(eval_df['label'], eval_df['prediction'])
    print("\n--- Classification Report ---")
    print(report)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(eval_df['label'], eval_df['prediction'], labels=classifier.categories)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classifier.categories, 
                yticklabels=classifier.categories)
    plt.title('Ticket Classification Confusion Matrix')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    
    plt.savefig('section_03_classifier/confusion_matrix.png')
    print("\n Confusion Matrix saved as 'section_03_classifier/confusion_matrix.png'")

if __name__ == "__main__":
    run_evaluation()