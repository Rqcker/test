import requests
import json
import csv
from time import sleep
# import zipfile
# import os

with open('KeywordsList.csv', 'r') as f:
  reader = csv.reader(f)
  keywords = []
  for row in reader:
    keywords.append(row[0])
# print(keywords)

for keyword in keywords:
  API_KEY = '0f5a7d4d-8d1b-4d65-8504-ee4688bad5e4'
  query = keyword
  num_articles = 1000  # Number of articles to retrieve
  articles_per_page = 10  # Number of articles per page

  # Calculate the number of pages to fetch
  num_pages = (num_articles - 1) // articles_per_page + 1

  # Create a list to store news data
  news_list = []

  # Loop through the pages to fetch news data
  for page in range(1, num_pages + 1):
      # Construct the request URL with the page parameter
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
  with open(f'{keyword}.json', 'w') as file:
      json.dump(news_list, file, indent=4)

  print(f"Total articles obtained: {len(news_list)}")
  print(f"News data saved as {keyword}.json")
  print('the Guardian API Bot Sleeping............')
  sleep(1) # quick break for the API limit :(

print("---Loop Done---")
print(f"Total number of JSON files created: {len(keywords)}")

# # Open a new ZIP file for writing
# with zipfile.ZipFile('keywords.zip', 'w') as zf:
#     # Loop through each JSON file in the output folder
#     for filename in os.listdir('/content/'):
#         # Check if the file ends with .json
#         if filename.endswith('.json'):
#             # Read the JSON file and add it to the ZIP archive
#             with open(os.path.join('/content/', filename), 'rb') as f:
#                 zf.writestr(filename, f.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
