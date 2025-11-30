import json
import re

# Define the list of companies as per the user's request
companies = [
    "Reliance Industries", "Reliance", "RIL", "Tata Consultancy Services",
    "TCS", "Tata Consultancy", "HDFC Bank", "HDFC", "ICICI Bank", "ICICI",
    "Bharti Airtel", "Airtel", "State Bank of India", "SBI", "Infosys",
    "Life Insurance Corporation", "LIC", "Hindustan Unilever", "HUL", "ITC"
]

# Construct the regex pattern
# (?i) for case-insensitive
# \b for word boundaries
pattern = r'(?i)\b(' + '|'.join(map(re.escape, companies)) + r')\b'
regex = re.compile(pattern)

input_file = 'IN-FINewsDataset.json'
output_file = 'art.json'

try:
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    print(f"Total records: {len(data)}")
    
    # Filter data
    # The user's Go code used 'companyRegex.MatchString(title)', so we match against the Title.
    filtered_data = [item for item in data if regex.search(item.get('Title', ''))]
    
    print(f"Filtered records: {len(filtered_data)}")
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)
        
    print("Done.")

except FileNotFoundError:
    print(f"Error: {input_file} not found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from {input_file}.")
except Exception as e:
    print(f"An error occurred: {e}")
