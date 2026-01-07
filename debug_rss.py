import requests
from bs4 import BeautifulSoup

def debug_rss():
    rss_url = "https://dailynewsegypt.com/feed/"
    try:
        response = requests.get(rss_url, timeout=5)
        soup = BeautifulSoup(response.content, features="xml")
        
        items = soup.find_all('item')
        print(f"Found {len(items)} items in feed.")
        
        for i, item in enumerate(items[:10]):
            title = item.title.text
            print(f"{i+1}. {title}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_rss()
