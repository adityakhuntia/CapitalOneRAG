import requests

url = "http://127.0.0.1:8000/chat"  # or "http://localhost:8000/chat"
payload = {
    "query": "My wheat has yellow leafs? Is it diseased ?What should I do?",
    "session_id": "farmer01",
    "language": "English",
    "region": "Punjab"
}

res = requests.post(url, json=payload)
print(res.json())
