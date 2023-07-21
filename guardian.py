import requests
import json

API_KEY = 'ec740d79-7d0f-4d73-87cf-458e7bcab9cf'
query = 'circular economy'
num_articles = 15000  # Number of articles to retrieve
articles_per_page = 10  # Number of articles per page

# Calculate the number of pages to fetch
num_pages = (num_articles - 1) // articles_per_page + 1

# Create a list to store news data
news_list = []

# Loop through the pages to fetch news data
for page in range(1, num_pages + 1):
    # Construct the request URL with the page parameter
    #url = f"https://content.guardianapis.com/search?q={query}&page={page}&api-key={API_KEY}"
    url = f"https://content.guardianapis.com/search?q={query}&show-fields=body&page={page}&api-key={API_KEY}"

    # Send the request
    response = requests.get(url)

    # Check the response status code
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data['response']['results']

        # Extract and store the news data
        for article in articles:
            title = article['webTitle']
            section = article['sectionName']
            date = article['webPublicationDate']
            link = article['webUrl']
            content = article['fields']

            # Create a dictionary to store the news article
            news_article = {
                'title': title,
                'section': section,
                'date': date,
                'link': link,
                'content': content
            }

            # Add the news article to the list
            news_list.append(news_article)

            # Print the news article
            print(f"Title: {title}")
            print(f"Section: {section}")
            print(f"Date: {date}")
            print(f"Link: {link}")
            print(f"Content: {content}")
            print('---')

    else:
        print(f"Request for page {page} failed")

# Save the news data as a JSON file
with open('news_data.json', 'w') as file:
    json.dump(news_list, file, indent=4)

print(f"Total articles obtained: {len(news_list)}")
print("News data saved as 'news_data.json'")
