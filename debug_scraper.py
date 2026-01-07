import requests
from bs4 import BeautifulSoup

URL = "https://egypt.gold-price-today.com/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

try:
    response = requests.get(URL, headers=headers)
    response.raise_for_status()
    print(f"Status Code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Print first few table rows to see structure
    rows = soup.find_all('tr')
    print(f"Found {len(rows)} rows.")
    for i, row in enumerate(rows[:10]):
        print(f"--- Row {i} ---")
        print(row.prettify())
        print(f"Text: {row.get_text().strip()}")

except Exception as e:
    print(f"Error: {e}")
