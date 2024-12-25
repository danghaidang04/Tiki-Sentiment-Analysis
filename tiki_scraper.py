import requests
import re
import pandas as pd


# Extract product ID from Tiki URL
def extract_product_id(url):
    match = re.search(r"-p(\d+)\.html", url)
    return match.group(1) if match else None


# Fetch reviews from Tiki
def scrape_tiki_reviews(product_id, max_reviews=100, reviews_per_page=20):
    reviews = []
    base_url = "https://tiki.vn/api/v2/reviews"
    page = 1

    while len(reviews) < max_reviews:
        params = {
            "product_id": product_id,
            "limit": reviews_per_page,  # Change this to 20 (or more) reviews per page
            "page": page
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        response = requests.get(base_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            review_data = data.get("data", [])

            if not review_data:  # No more reviews available
                break

            for review in review_data:
                content = review.get("content", "")
                rating = review.get("rating", 0)
                date = review.get("timeline", {}).get("review_created_date", "N/A")
                user = review.get("created_by", {}).get("name", "Anonymous")

                if content:  # Only include non-empty reviews
                    reviews.append({
                        "User": user,
                        "Rating": rating,
                        "Review": content,
                        "Date": date
                    })

                # Stop if we've reached the maximum reviews
                if len(reviews) >= max_reviews:
                    break

            page += 1  # Move to the next page if needed

        else:
            print(f"Failed to fetch reviews: {response.status_code}")
            break

    # Trim the reviews to the maximum limit if necessary
    return pd.DataFrame(reviews[:max_reviews])
