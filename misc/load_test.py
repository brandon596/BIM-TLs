import asyncio
import httpx
import os

async def test_semantic_search(query="what is the project browser?", num=10):
    url = "http://localhost:5000/api/semantic_search"  # Replace with your actual URL
    header = {"Authorization": os.environ.get("API_KEY")}  # Replace with your actual headers
    body = {"query": query}  # Replace with your actual body structure

    async with httpx.AsyncClient() as client:
        for x in range(num):
            response = await client.post(url, headers=header, json=body)
            data = response.json()  # Parse the JSON response
            assert data.get("Title", [None])[0] == "Open Project Browser and Properties", \
                f"Test failed at iteration {x+1}"



if __name__ == "__main__":
    asyncio.run(test_semantic_search(num=200))