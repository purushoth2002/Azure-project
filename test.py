import numpy as np
import os
from flask import Flask, app,request,render_template, redirect, url_for,jsonify
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
import requests
import tensorflow as tf
from PIL import Image
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from transformers import pipeline
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import requests
app=Flask(__name__)


model_name = "t5-small"  # You can also use "t5-base" or "t5-large" for better performance
# Load the pre-trained model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning",from_tf=True)
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

tokenizer_for_summarization = T5Tokenizer.from_pretrained(model_name)
summarization_model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_image(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    return pixel_values

def generate_caption(image_path, max_length=16, num_beams=4):
    pixel_values = preprocess_image(image_path)
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


def summarize_text(text, max_length=150, min_length=40, length_penalty=2.0, num_beams=4):
    # Prepare the text for the T5 model
    input_text = "summarize: " + text
    input_ids = tokenizer_for_summarization.encode(input_text, return_tensors="pt", max_length=512, truncation=True)


    # Generate summary
    summary_ids = summarization_model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=num_beams,
        early_stopping=True
    )

    # Decode and return the summary
    summary = tokenizer_for_summarization.decode(summary_ids[0], skip_special_tokens=True)
    return summary



#default home page or route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index')
def inde1():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/clients',methods=['GET','POST'])
def clients():
    if(request.method == 'GET'):
        return render_template('clients.html')
    if(request.method =='POST'):

        text = request.form['text-input']
        summary = summarize_text(text)
        tokens = summary.split('.')
        tokens = [token.strip() for token in tokens]
        res = [token.capitalize() for token in tokens]
        return render_template('clients.html',summary='. '.join(res),text=text)

@app.route('/ourwork',methods=['GET','POST'])
def ourwork():
    if(request.method=='GET'):
        return render_template('ourwork.html')
    if(request.method =='POST'):
        f = request.files['image']
        path=f'static/uploads/{f.filename}'
        f.save(path)
        caption = generate_caption(path)
        caption = caption.capitalize()
        return render_template('ourwork.html',path=path,caption=caption)




@app.route('/contact')
def contact():
    return render_template('contact.html')

        



""" Running our application """
if __name__ == "__main__":
    app.run(debug =True, host = "0.0.0.0", port = 8080)
