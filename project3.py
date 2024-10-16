import requests
from bs4 import BeautifulSoup
import csv
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Google API setup
API_KEY = 'your API key'  # Replace with your API key
SEARCH_ENGINE_ID = 'search engine ID'  # Replace with your search engine ID

# URL to scrape
url = 'Enter you article url'

def scrape_article(url):
    """Scrape the article content from the given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Safely get the title
        title = soup.title.get_text().strip() if soup.title else 'No Title'
        
        paragraphs = soup.find_all('p')
        article_text = ' '.join(p.get_text().strip() for p in paragraphs)
        return title, article_text
    return None, ""


def get_similar_articles(search_query):
    """Fetch similar articles using the Google Custom Search API."""
    search_url = f'https://www.googleapis.com/customsearch/v1?q={search_query}&key={API_KEY}&cx={SEARCH_ENGINE_ID}'
    search_response = requests.get(search_url)
    
    if search_response.status_code == 200:
        return [item['link'] for item in search_response.json().get('items', [])]
    return []

# Scrape the original article
title, original_article = scrape_article(url)

# Save the original article summary to CSV
with open('article_summary.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Title', 'Summary'])
    writer.writerow([title, original_article])

print("Data saved to article_summary.csv")

# Search for similar articles
search_query = title  # You can modify this to use a key phrase or summary
similar_article_urls = get_similar_articles(search_query)

# Scrape similar articles and extract unique points
similar_articles = []
for article_url in similar_article_urls:
    _, article_content = scrape_article(article_url)
    similar_articles.append(article_content)

# Extract unique points using TF-IDF and cosine similarity
if similar_articles:
    vectorizer = TfidfVectorizer().fit_transform([original_article] + similar_articles)
    vectors = vectorizer.toarray()

    cosine_sim = cosine_similarity(vectors)
    unique_points = defaultdict(list)

    for i in range(1, len(vectors)):  # Start from 1 to skip the original article
        if cosine_sim[0][i] < 0.5:  # Threshold to determine uniqueness
            unique_points[0].append(similar_articles[i - 1])  # Adjust index

    # Display unique points from similar articles
    for index, unique in unique_points.items():
        print(f"\nUnique points from Similar Articles:")
        for point in unique:
            print(point)
else:
    print("No similar articles found.")


