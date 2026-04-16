import re

import requests

API_URL = "http://localhost:3000/api/v1/prediction/616c4ca5-330c-4068-bbd0-1d8fa9889449"


def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()


question = "Welche Hersteller von Steuererklärungsprogrammen unterstützen ELSTER?"

output = query(
    {
        "question": question,
    }
)
# r for raw string
pattern = r".*?(?=<ui)"
match = re.search(pattern, output["text"], re.DOTALL)
if match:
    extracted_text = match.group(0).strip()
    print("Extracted Text:")
    print(extracted_text)
