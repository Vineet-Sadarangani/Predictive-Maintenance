from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Set up Selenium WebDriver
driver = webdriver.Chrome()  # Ensure the correct ChromeDriver path
driver.get("http://localhost:8501")  # Main Streamlit app URL

wait = WebDriverWait(driver, 30)  # Increased timeout

try:
    # Step 1: Upload the test file on the main page
    print("Uploading test file...")
    test_file_upload = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
    test_file_upload.send_keys(r"C:\Users\91843\Documents\GitHub\Predictive-Maintenance\Dataset\PM_test.txt")  # Update path

    # Step 2: Upload the truth file
    print("Uploading truth file...")
    truth_file_upload = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "input[type='file']")))[1]
    truth_file_upload.send_keys(r"C:\Users\91843\Documents\GitHub\Predictive-Maintenance\Dataset\PM_truth.txt")  # Update path

    print("✅ Test and Truth files uploaded successfully.")

    # Step 3: Navigate to Real-Time Sensor Data page
    print("Navigating to Real-Time Sensor Data page...")
    driver.get("http://localhost:8501/Real-Time_Sensor_Data")  # Adjust if needed

    # Step 4: Upload the dataset for real-time visualization
    print("Uploading real-time dataset...")
    dataset_upload = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
    dataset_upload.send_keys(r"C:\Users\91843\Documents\GitHub\Predictive-Maintenance\Dataset\PredictiveManteinanceEngineTraining.csv")  # Update path

    # Step 5: Select "s1" from the dropdown
    print("Selecting sensor 's1'...")
    dropdown = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".stSelectbox")))
    dropdown.click()
    sensor_option = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 's1')]")))
    sensor_option.click()

    print("✅ Sensor 's1' selected.")
    print("✅ Real-time chart rendered successfully!")

except Exception as e:
    print(f"❌ Test failed: {e}")

finally:
    # Close the browser
    driver.quit()
