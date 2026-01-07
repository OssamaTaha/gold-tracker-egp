import requests
from bs4 import BeautifulSoup

URL = "https://egypt.gold-price-today.com/"
try:
    response = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    
    links = soup.find_all('a')
    print(f"Found {len(links)} links.")
    for link in links:
        href = link.get('href')
        text = link.get_text().strip()
        if href and ("history" in href or "archive" in href or "date" in href or "prev" in href):
            print(f"Candidate: {text} -> {href}")
            
except Exception as e:
    print(e)
