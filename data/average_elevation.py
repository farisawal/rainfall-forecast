# pip install selenium pandas openpyxl

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import re
from time import sleep

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--headless=new") # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--enable-unsafe-swiftshader")

url = "https://en-gb.topographic-map.com/"

siteList = "Store List for ESG.xlsx"
try:
    site = pd.read_excel(siteList)
    site['elevation'] = None
except Exception as e:
        print(f"Error reading Excel file: {e}")
        exit()

driver = webdriver.Chrome(options=chrome_options)

for index, coord in site.iterrows():    
    lat, lon = coord.iloc[1], coord.iloc[2]
    
    if pd.isna(lat) or pd.isna(lon):
        print(f"\nSkipping row {index + 1} due to missing latitude or longitude.")
        site.at[index, 'elevation'] = "Invalid Coordinates"
        continue

    try:
        print(f"\rProcessing coordinate {index + 1}/{len(site)}: {lat},{lon}")
        driver.get(url)
        # Wait for the search input field to be visible and clickable
        search_input = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.NAME, "query"))
        )
        # Scroll into view and clear the input field
        driver.execute_script("arguments[0].scrollIntoView(true);", search_input)
        search_input.clear()

        # Input the coordinates and submit
        search_input.send_keys(f"{lat},{lon}")
        search_input.send_keys(Keys.RETURN)

        # Wait for elevation data
        elevation = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "about"))
        )

        # Extract average elevation
        match = re.search(r"Average elevation: ([\d,]+)", elevation.text)
        if match:
            average_elevation = match.group(1)
            print(f"Extracted Average Elevation: {average_elevation}")
            site.at[index, 'elevation'] = average_elevation

        else:
            print("Average elevation not found in the text.")
            site.at[index, 'elevation'] = "N/A"

    except Exception as e:
        print(f"An error occurred while processing {lat},{lon}: {e}")

# Close the WebDriver
driver.quit()

try:
    site.to_excel(f"{siteList}", index=False)
    print("\nScraping complete!")
    print(f"Total coordinates processed: {len(site)}")
    print(f"Successful elevations retrieved: {site['elevation'].notna().sum()}")
    print(f"Failed retrievals: {site['elevation'].isna().sum()}")
    print(f"\nResults saved to '{siteList}'")
except Exception as e:
    print(f"Error saving Excel file: {e}")