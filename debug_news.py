import requests
from bs4 import BeautifulSoup

def debug_google_rss():
    url = "https://news.google.com/rss/search?q=Egypt+Gold+Price+Economy&hl=en-EG&gl=EG&ceid=EG:en"
    
    print("Attempt 1: No Headers")
    try:
        r = requests.get(url, timeout=5)
        print(f"Status: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nAttempt 2: With User-Agent")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        r = requests.get(url, headers=headers, timeout=5)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, features="xml")
            items = soup.find_all('item')
            print(f"Found {len(items)} items.")
            print(f"Title 1: {items[0].title.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_google_rss()
