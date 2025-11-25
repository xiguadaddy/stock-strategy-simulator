from urllib.parse import unquote

url = "https://m.aastocks.com/sc/usq/quote/stock-news-content.aspx%3Fsymbol%3DTSLA%26id%3DNOW.1484291%26source%3DAAFN"
decoded_url = unquote(url)
print(f"Original: {url}")
print(f"Decoded: {decoded_url}")
