# MathHelper - A Webapp for solving handwritten equations using CNN
The current version is capable of handling one-dimensional arithmetic expressions, including basic addition, subtraction, multiplication, division, sine, cosine, tangent, logarithm, factorial, and pi. We aim to implement recognition for more complex mathematical calculations in the future, such as exponentiation and binomial polynomials.

## Dataset
dataset from (https://www.kaggle.com/xainano/handwrittenmathsymbols) + (http://yann.lecun.com/exdb/mnist/)

## Overview
![image](https://github.com/Tristaaaa/MathHelper/blob/main/test/overview.png)

### app.py(FRONTEND)
Using Flask to build a webpage, implementing an interactive sketchpad with p5.js as an interface between the user and backend calculations.

### cnn.py(BACKEND)
The incoming photos will undergo feature extraction, followed by segmentation to separate digits and symbols. Each segment will then be fed into a pre-trained CNN model for prediction. Subsequently, the predicted strings will be evaluated using the eval function to compute the answers, which will be returned to the frontend.

### model
pretrained model of the CNN

### training model and comparison
- classifier_train.ipynb
- classifier_validate.ipynb
- prediction.ipynb
- train_final.ipynb

## Test result
<p float="left">
    <img src="https://github.com/Tristaaaa/MathHelper/blob/main/test/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-06-10%20191449.png" alt="1" width="400" />
    <img src="https://github.com/Tristaaaa/MathHelper/blob/main/test/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-06-10%20185114.png" alt="2" width="400" /> 
</p>

<p float="left">
    <img src="https://github.com/Tristaaaa/MathHelper/blob/main/test/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-06-10%20185806.png" alt="1" width="400" />
    <img src="https://github.com/Tristaaaa/MathHelper/blob/main/test/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-06-10%20185359.png" alt="2" width="400" /> 
</p>
