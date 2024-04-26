import json
import uvicorn
import pickle
from fastapi import FastAPI
from UrlData import UrlData, DomainData
from Utils import getTypoSquattedDomains
from API import get_prediction
from fastapi.middleware.cors import CORSMiddleware
from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import zipfile
import os
from fastapi import FastAPI
import cv2
from skimage.metrics import structural_similarity
import whois
import requests
from bs4 import BeautifulSoup
import zipfile
import base64
from langchain.llms import cohere


app = FastAPI(debug=True)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------

# load the LightGBM classifier using pickle
print("Loading the model...")
with open("./lightgbm_classifier.pkl", "rb") as file:
    clf = pickle.load(file)


@app.post("/generate")
async def generate(prompt: str):
    cohere_api_key = "daFVckVElGhcDTwrfw7f6zvf1xivC4F3YTZE2jpf"
    llm = cohere.Cohere()
    

@app.get("/get_new_domains")
def get_new_domains():
    r = requests.get("https://whoisds.com/newly-registered-domains")
    soup = BeautifulSoup(r.content, "html5lib")
    soup = soup.find_all("table", attrs={"class": "table table-bordered"})[0]
    link = soup.find_all("a")[1].get("href")

    zip_file = requests.get(link)

    with open("domain_0.zip", "wb") as f:
        f.write(zip_file.content)

    all_links = []

    for file in os.listdir():
        if file.endswith(".zip"):
            file_path = os.path.join("./", file)

            with zipfile.ZipFile(file_path) as z:
                for txt_file in z.namelist():
                    if txt_file.endswith(".txt"):
                        with z.open(txt_file) as f:
                            links = f.readlines()
                            all_links.extend(
                                [link.decode("utf-8").strip() for link in links]
                            )

    return all_links[:200]


@app.post("/post_images")
async def get_images(url1: str, url2: str):
    options = webdriver.ChromeOptions()
    options.headless = True
    driver = webdriver.Chrome(options=options)

    driver.get(url1)

    S = lambda X: driver.execute_script(f"return document.body.parentNode.scroll{X}")
    driver.set_window_size(S("Width"), S("Height"))
    driver.find_element(By.TAG_NAME, "body").screenshot("web_screenshot1.png")

    driver.quit()

    options = webdriver.ChromeOptions()
    options.headless = True
    driver = webdriver.Chrome(options=options)

    driver.get(url2)

    S = lambda X: driver.execute_script(f"return document.body.parentNode.scroll{X}")
    driver.set_window_size(S("Width"), S("Height"))
    driver.find_element(By.TAG_NAME, "body").screenshot("web_screenshot2.png")

    driver.quit()

    first1 = cv2.imread("D:\\Phishr-API\\web_screenshot1.png")
    second1 = cv2.imread("D:\\Phishr-API\\web_screenshot2.png")

    first = cv2.resize(first1, (989, 744))
    second = cv2.resize(second1, (989, 744))

    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, diff = structural_similarity(first_gray, second_gray, full=True)
    print("Similarity Score: {:.3f}%".format(score * 100))

    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(first, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(second, (x, y), (x + w, y + h), (36, 255, 12), 2)

    retval, buffer_img1 = cv2.imencode("first.jpg", first)
    img1_base64 = base64.b64encode(buffer_img1).decode("utf-8")

    retval, buffer_img2 = cv2.imencode("second.jpg", second)
    img2_base64 = base64.b64encode(buffer_img2).decode("utf-8")

    retval, buffer_img3 = cv2.imencode("s1.jpg", first1)
    img3_base64 = base64.b64encode(buffer_img3).decode("utf-8")

    retval, buffer_img4 = cv2.imencode("s2.jpg", second1)
    img4_base64 = base64.b64encode(buffer_img4).decode("utf-8")

    # Return base64 strings
    return {
        "image1": img1_base64,
        "image2": img2_base64,
        "image3": img3_base64,
        "image4": img4_base64,
        "ssim_score": score * 100,
    }


@app.post("/whois")
async def whoisfunc(domain: str):
    return whois.whois(domain)


@app.post("/predict")
def predict(data: UrlData):
    # convert to dictionary
    data = data.dict()

    # the key has same name as you put in class
    url = data["url"]

    # predict price using ML model
    prediction = get_prediction(url, clf)
    print("Predicted Probability : ", prediction)

    # always return the output as dictionary/json.
    prediction = {"prediction": prediction}
    return prediction


@app.post("/get_typesquatted_urls")
def get_similar_urls(data: DomainData):
    # convert to dictionary
    data = data.dict()

    # the key has same name as you put in class
    url = data["url"]
    max_num = data["max_num"]

    if max_num <= 0:
        max_num = 20

    # result
    output = getTypoSquattedDomains(url, max_num)
    print("API OUTPUT : ", output)
    output = {"output": output}

    # Convert the output dictionary to JSON-compatible format
    output_dict = json.loads(json.dumps(output, default=str))
    return output_dict


if __name__ == "__main__":
    uvicorn.run(app)
