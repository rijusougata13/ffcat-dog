from flask import request
from flask import jsonify
from flask import Flask
from flask import render_template
from flask import request,redirect
import cv2
import tensorflow as tf
from keras.models import load_model
import keras
from tensorflow import keras
from keras.applications.resnet50 import preprocess_input, decode_predictions
import os
from keras.preprocessing import image
import numpy as np


categories=['Dog','Cat']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static/images')
@app.route('/',methods=['POST','GET'])
def hello():
 	
    if request.method=='POST':
       
        file = request.files['file']
        
        file_path=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        image1=img
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        model1 = tf.keras.models.load_model('dogcat_model_bak.h5')
        prediction=model1.predict([x])
        proba=model1.predict_proba([x])
        #return(categories[int(prediction[0][0])])
       
        return render_template('submit.html',image=file_path,value=categories[int(prediction[0][0])])

    else:
        return render_template("hello.html")
if __name__ == '__main__':
    app.run(debug=True)