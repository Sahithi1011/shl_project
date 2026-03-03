from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

chrome_options = Options()
chrome_options.add_argument("--start-maximized")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

base_url = "https://www.shl.com/products/product-catalog/?type=1&start="

all_data = []
unique_links = set()

start = 0
page_size = 12   # SHL shows 12 per page

while True:
    url = base_url + str(start)
    print("Opening:", url)
    driver.get(url)
    time.sleep(6)

    cards = driver.find_elements(By.XPATH, "//a[contains(@href, '/products/')]")
    
    if len(cards) == 0:
        print("No more cards found. Stopping.")
        break

    print("Found cards:", len(cards))

    new_items = 0

    for card in cards:
        link = card.get_attribute("href")
        title = card.text.strip()

        if link and title and link not in unique_links:
            unique_links.add(link)
            all_data.append({
                "assessment_name": title,
                "description": "",
                "url": link,
                "test_type": ""
            })
            new_items += 1

    print("New items added:", new_items)

    if new_items == 0:
        break

    start += page_size

driver.quit()

df = pd.DataFrame(all_data)
print("Total assessments collected:", len(df))

df.to_csv("shl_assessments.csv", index=False)
print("CSV saved successfully!")