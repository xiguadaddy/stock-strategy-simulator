from GoogleNews import GoogleNews

googlenews = GoogleNews(lang='en', period='7d')
googlenews.search('Tesla')
results = googlenews.result()
if results:
    print(f"First link: {results[0].get('link')}")
else:
    print("No results")
