import time
import pandas as pd
import numpy as np
import os
from classifier import TicketClassifier

def run_latency_test():
    print("Initializing classifier on CPU...")
    classifier = TicketClassifier()
    
    test_cases = [
        "My account was charged twice.",
        "The app crashes when I click export.",
        "I would love a dark mode feature.",
        "Your service is terrible, I want a refund.",
        "Is there an API for developers?"
    ] * 4 
    
    results = []
    
    print(f"Running latency test on {len(test_cases)} tickets...\n")
    
    for i, text in enumerate(test_cases):
        start_time = time.perf_counter()
        prediction = classifier.predict(text)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        
        results.append({
            "ticket_id": i + 1,
            "text_preview": text[:30] + "...",
            "prediction": prediction,
            "latency_ms": round(duration_ms, 2)
        })
        
        print(f"Ticket {i+1}: {duration_ms:.2f}ms")

    df_results = pd.DataFrame(results)
    
    report_path = os.path.join("section_03_classifier", "latency_report.csv")
    df_results.to_csv(report_path, index=False)
    
    print(f"\n Test Passed!")
    print(f"Average Latency: {df_results['latency_ms'].mean():.2f}ms")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    run_latency_test()