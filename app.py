import tensorflow as tf
import base64
import json

from flask import Flask, render_template, request
from cnn import ConvolutionalNeuralNetwork
from io import BytesIO
from PIL import Image
import os
from werkzeug.utils import secure_filename
import cv2
import math

app = Flask(__name__)

def calculate_operation(operation):
    def operate(fb, sb, op):
        if operator == '+':
            result = int(first_buffer) + int(second_buffer)
        elif operator == '-':
            result = int(first_buffer) - int(second_buffer)
        elif operator == 'x':
            result = int(first_buffer) * int(second_buffer)
        elif operator == '/':
            result = int(first_buffer) / int(second_buffer)
        return result

    if not operation or not operation[0].isdigit():
        return -1

    operator = ''
    first_buffer = ''
    second_buffer = ''

    for i in range(len(operation)):
        if operation[i].isdigit():
            if len(second_buffer) == 0 and len(operator) == 0:
                first_buffer += operation[i]
            else:
                second_buffer += operation[i]
        else:
            if len(second_buffer) != 0:
                result = operate(first_buffer, second_buffer, operator)
                first_buffer = str(result)
                second_buffer = ''
            operator = operation[i]

    result = int(first_buffer)
    if len(second_buffer) != 0 and len(operator) != 0:
        result = operate(first_buffer, second_buffer, operator)

    return result
    
UPLOAD_FOLDER = 'home\trista\下載'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #operation = BytesIO(base64.urlsafe_b64decode(request.form['operation']))
    CNN = ConvolutionalNeuralNetwork()
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 
                                   filename),cv2.IMREAD_GRAYSCALE)
    pred_oper, operation = CNN.predict(img)

    return json.dumps({
        'operation': pred_oper,
        'solution': eval(operation)
    })

@app.route('/predict', methods=['POST'])
def predict():
    operation = BytesIO(base64.urlsafe_b64decode(request.form['operation']))
    Image.open(operation).save('aux.png')
    img = cv2.imread('aux.png',cv2.IMREAD_GRAYSCALE)
    os.remove('aux.png')
    CNN = ConvolutionalNeuralNetwork()
    pred_oper, operation = CNN.predict(img)

    return json.dumps({
        'operation': pred_oper,
        'solution': eval(operation)
    })
    



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
