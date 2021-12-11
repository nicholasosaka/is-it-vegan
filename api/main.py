print(f"Importing dependencies")
from flask import Flask, request, jsonify
import cv2
import pytesseract
import time
import numpy as np 
from joblib import load
from PIL import Image
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


print("Finding best model for classification")
best_model_name_path = "../models/model.txt"
best_model = None
with open(best_model_name_path, "r") as f:
    best_model = f.readline()

print(f"Loading dataset")
with open('../data/item_info.pkl', 'rb') as pickle_file:
    item_info = pickle.load(pickle_file)

all_data = pd.DataFrame(columns=["ingredients", "is_vegan"])



def clean_text(s):
    return re.sub(r'[^a-zA-Z]', ' ', s).lower().split()

for item_id in item_info:
    all_data = all_data.append({
        "ingredients": clean_text(item_info[item_id]['ingredients']),
        "is_vegan": item_info[item_id]["is_vegan"]
    }, ignore_index=True)

all_data['ingredients'] = all_data['ingredients'].apply(lambda l: " ".join(l))
X = all_data['ingredients']
y = all_data['is_vegan']

print(f"Fitting TF-IDF pipeline to dataset")
pipeline = Pipeline([
    ('countvec', CountVectorizer(
                    lowercase=False,
                    tokenizer=lambda x:x,
                    ngram_range=(2,5),
                    analyzer="word"
                )
    ),
    ('tf-idf', TfidfTransformer(
                    norm='l2',
                )
    )
]).fit(X)


print(f"Loading {best_model}")
clf = load(f"../models/{best_model}.joblib")


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# CONF_THRESHOLD = 0.9

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def is_it_vegan():
    response = jsonify({"status": "OK"})
    response.status_code = 200
    return response


@app.route("/process", methods=["POST"])
def process():
    im_file = request.files['image']

    if im_file.filename == "":
        response = jsonify({"status": "No file uploaded"})
        response.status_code = 400
        return response
    
    upload_file_name = str(time.time()).split('.')[0] + '.png'
    filename = UPLOAD_FOLDER + '/' + upload_file_name
    im_file.save(filename)
    img = Image.open(im_file.stream)

    TAG = upload_file_name.split(".")[0]

    print(f"[{TAG}] Image received.")

    payload = {
        "file_name": upload_file_name,
        "image_size": {
            "x": img.width,
            "y": img.height
        }
    }

    # ocr
    ocr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    ocr_img = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)
    ocr_text = pytesseract.image_to_string(ocr_img)
    ocr_text = ocr_text.strip()

    if len(ocr_text) > 0:
        print(f"[{TAG}] Text found.")
        cleaned_ocr_text = " ".join(clean_text(ocr_text))
        payload["ocr"] = {
            "text": ocr_text,
            "cleaned_text": cleaned_ocr_text
        }


        print(f"[{TAG}] Running inference")
        X_sample = pipeline.transform([cleaned_ocr_text])
        pred = clf.predict(X_sample)

        prediction_as_str = "vegan" if bool(pred) else "not vegan"
        payload["prediction"] = prediction_as_str
        
    else:
        print(f"[{TAG}] No text found.")
        response = jsonify(payload)
        response.status_code = 200
        payload["status"] = "No text found."
        return response


    payload["status"] = "Success."
    response = jsonify(payload)
    response.status_code = 200
    return response