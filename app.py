from flask import request
from flask import jsonify
from flask import Flask
from flask import render_template
from flask import request,redirect
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import os
from keras.preprocessing import image
import numpy as np


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static/images')


@app.route('/',methods=['POST','GET'])
def hello():
    if request.method=='POST':
        file = request.files['file']
        file_path=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(file_path)
        img = image.load_img(file_path, target_size=(128, 128))
        image_classifier = load_model('cat_dog_classifier_v1.h5')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        class_labels = {0: 'Dog', 1: 'Cat'}
        prediction = image_classifier.predict_classes(x)
        percent_values = prediction.tolist()   
        guess = class_labels[prediction[0][0]] 
        print(percent_values)  
        return render_template('submit.html',image=file_path,value=guess)

    else:
        return render_template("hello.html")

if __name__ == '__main__':
    app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=process.env.PORT )