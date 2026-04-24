import os
import csv
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables and initialize client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Categories as defined in the task
CATEGORIES = ["billing", "technical_issue", "feature_request", "complaint", "other"]

def generate_tickets(category, count=200):
    """
    Calls the LLM to generate synthetic support tickets.
    """
    print(f"Generating {count} tickets for category: {category}...")
    
    prompt = f"""
    Generate {count} unique customer support tickets for the category: '{category}'.
    Requirements:
    1. Vary the length (one sentence to a paragraph).
    2. Vary the sentiment (polite, frustrated, confused).
    3. Use realistic scenarios related to a SaaS product.
    4. Format as a simple list with one ticket per line.
    No numbers, bullet points, or empty lines.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant that generates synthetic data for machine learning training."},
                      {"role": "user", "content": prompt}],
            temperature=0.8
        )
        
        # Split by newline and clean up the list
        content = response.choices[0].message.content.strip()
        tickets = content.split('\n')
        
        # Return a list of dictionaries
        return [{"text": t.strip(), "label": category} for t in tickets if len(t.strip()) > 5]
    
    except Exception as e:
        print(f"Error generating data for {category}: {e}")
        return []

def save_to_csv(data, filename="section_03_classifier/data/training_tickets.csv"):
    """
    Saves the list of dictionaries to a CSV file.
    """
    if not data:
        print("No data found to save.")
        return

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    print(f"Successfully saved {len(data)} tickets to {filename}")

if __name__ == "__main__":
    all_data = []
    
    for cat in CATEGORIES:
        # Generate tickets for the category
        category_tickets = generate_tickets(cat, 200)
        
        # Use .extend to keep the list flat (list of dicts)
        all_data.extend(category_tickets)
    
    # Final save
    save_to_csv(all_data)