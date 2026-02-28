from playwright.sync_api import sync_playwright

def verify_map():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print("Navigating to map.html...")
            page.goto("http://localhost:8000/map.html")

            print("Waiting for leaflet container...")
            # Wait for the map to load
            page.wait_for_selector(".leaflet-container")

            print("Taking screenshot...")
            page.screenshot(path="map_screenshot.png")
            print("Screenshot saved to map_screenshot.png")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_map()
