import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


TARGET_URL = "https://www.keysso.net/calls"


def looks_interesting(url: str) -> bool:
    lower = url.lower()

    keywords = [
        "call",
        "calls",
        "calllog",
        "ajax",
        "api",
        "incident",
        "cad",
        "service",
        "log",
        "complaint",
    ]

    return any(k in lower for k in keywords)


def main():
    options = Options()

    # Keep browser visible first. After it works, you can enable headless.
    # options.add_argument("--headless=new")

    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,1000")

    # Enable Chrome network/performance logging.
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(options=options)

    try:
        print(f"Opening: {TARGET_URL}")
        driver.get(TARGET_URL)

        # Wait for JavaScript/AJAX to load.
        time.sleep(10)

        print("\n==============================")
        print("Candidate network request URLs")
        print("==============================")

        found_urls = set()

        logs = driver.get_log("performance")

        for entry in logs:
            try:
                message = json.loads(entry["message"])["message"]
            except Exception:
                continue

            method = message.get("method", "")

            if method != "Network.requestWillBeSent":
                continue

            request = message.get("params", {}).get("request", {})
            url = request.get("url", "")

            if not url:
                continue

            if looks_interesting(url):
                if url not in found_urls:
                    found_urls.add(url)
                    print(url)

        print("\n==============================")
        print("Fetch / XHR resources")
        print("==============================")

        resources = driver.execute_script("""
            return performance.getEntriesByType('resource').map(e => ({
                name: e.name,
                initiatorType: e.initiatorType,
                transferSize: e.transferSize,
                duration: e.duration
            }));
        """)

        for r in resources:
            url = r.get("name", "")
            initiator = r.get("initiatorType", "")

            if initiator in ("fetch", "xmlhttprequest") or looks_interesting(url):
                print(f"{initiator:15} {url}")

        print("\n==============================")
        print("Visible page text sample")
        print("==============================")

        body_text = driver.find_element(By.TAG_NAME, "body").text
        print(body_text[:4000])

    finally:
        driver.quit()


if __name__ == "__main__":
    main()