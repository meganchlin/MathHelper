# MathHelper - A Webapp for solving handwritten equations using CNN
目前的版本能夠處理的一維算術表達式包括基本的加減乘除、 sin、cos、tan、 log、factorial 、pi，之後期待能夠實作出識別更加複雜的數學計算(次方、二元多次式)

## Dataset
dataset from (https://www.kaggle.com/xainano/handwrittenmathsymbols) + (http://yann.lecun.com/exdb/mnist/)

## Overview
![image](https://github.com/Tristaaaa/MathHelper/blob/main/test/overview.png)

### app.py(FRONTEND)
以 flask 建立網頁，透過 p5.js 實作互動式 sketchpad，作為使用者和後端計算之間的介面

### cnn.py(BACKEND)
將輸入進來的照片進行 feature extraction 經由 segmentation 分離數字和符號，再各別進入 train 好的 CNN model 進行預測，然後將前面得到的算式利用 eval 函式將預測的 string 算出答案再回傳至前端

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
