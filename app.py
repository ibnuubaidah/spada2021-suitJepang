from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('model_spada.h5')

class_dict = {0: 'Batu', 1: 'Gunting', 2:'Kertas'}

def predict_label(img_path):
    #loaded_img = load_img(img_path, target_size=(300,300))
    #img_array = img_to_array(loaded_img) / 255.0
    #img_array = expand_dims(img_array, 0)
    #predicted_bit = np.round(model.predict(img_array)[0][0]).astype('int')
    #return class_dict[predicted_bit]

    img = load_img(img_path, target_size=(300,300))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images)
    prediksi = max(classes[0])

    if classes[0,0]==prediksi:
        return class_dict[0]
    elif classes[0,1]==prediksi:
        return class_dict[1]
    else:
        return class_dict[2]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)