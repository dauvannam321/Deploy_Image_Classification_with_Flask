import os
import torch
from vgg16 import VGG16
from utils import *

import flask
from flask import Flask, jsonify, request, render_template


# Config
model_file = "models\\vgg16_weights.pth"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

# Load model
# Load the saved weights into the model
loaded_model = torch.load(model_file, map_location=torch.device('cpu'))

model = VGG16(2)
model.load_state_dict(loaded_model)
model.eval()


@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method == "POST":
        # get uploaded image file if it exists
		file = request.files['image']
		if not file: 
			return render_template('index.html', label="No file")
		path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
		file.save(path_to_save)
		print(path_to_save)
		
		image = imgToTensor(path_to_save)
		result = predict(model, image)


		return render_template('index.html', label=result, uploaded_image=path_to_save)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port = 8080)
