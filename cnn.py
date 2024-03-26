import pandas as pd
import numpy as np
import cv2
import os

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from os.path import isfile, join
from keras import backend as K
from os import listdir
from PIL import Image
from skimage.morphology import skeletonize
import math

labelencoder = ['!', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'cos', 'div', 'log', 'pi', 'sin', 'times']


def extract_imgs(location,img):
    x,y,w,h=location
    # 只提取方框内的東西
    extracted_img = img[y:y + h, x:x + w]
    # 將提取出的img resize成IMG_SIZE*IMG_SIZE大小的binary
    #black = np.zeros((96, 96), np.uint8)
    white = np.full((45, 45), 255, np.uint8)
    if (w > h):
        res = cv2.resize(extracted_img, (45, (int)(h * 45 / w)), interpolation=cv2.INTER_AREA)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        white[d:res.shape[0] + d, 0:res.shape[1]] = res
    else:
        res = cv2.resize(extracted_img, ((int)(w * 45 / h), 45), interpolation=cv2.INTER_AREA)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        #black[0:res.shape[0], d:res.shape[1] + d] = res
        white[0:res.shape[0], d:res.shape[1] + d] = res

    extracted_img = ~white
    #extracted_img = skeletonize(~white/255)*255
    #extracted_img = skeletonize(black)
    #extracted_img = np.logical_not(extracted_img)
    return extracted_img

class ConvolutionalNeuralNetwork:
    def __init__(self):
        if os.path.exists('model/model_weights.h5') and os.path.exists('model/model.json'):
            self.load_model()
        else:
            self.train_model()
            self.export_model()

    def load_model(self):
        print('Loading Model...')
        model_json = open('model/model.json', 'r')
        loaded_model_json = model_json.read()
        model_json.close()
        loaded_model = model_from_json(loaded_model_json)

        print('Loading weights...')
        loaded_model.load_weights("model/model_weights.h5")

        self.model = loaded_model

    def train_model(self):
        csv_train_data = pd.read_csv('model/train_data.csv', index_col=False)

        # The last column contain the results
        y_train = csv_train_data[['784']]
        csv_train_data.drop(csv_train_data.columns[[784]], axis=1, inplace=True)
        csv_train_data.head()

        y_train = np.array(y_train)

        x_train = []
        for i in range(len(csv_train_data)):
            x_train.append(np.array(csv_train_data[i:i+1]).reshape(1, 28, 28))
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, (-1, 28, 28, 1))

        # Train the model.
        print('Training model...')
        self.model.fit(
          x_train,
          to_categorical(y_train, num_classes=13),
          epochs=10,
          batch_size=200,
          shuffle=True,
          verbose=1
        )

    def export_model(self):
        model_json = self.model.to_json()
        with open('model/model.json', 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights('model/model_weights.h5')
            
    def predict(self, img):

        if img is not None:
            #mask_size = 11
            #mask = np.ones((mask_size, mask_size), np.float32)/ (mask_size ** 2)
            #smoothing_img = cv2.filter2D(img,-1,mask)

            (thresh, blackAndWhiteImage) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


            size = np.shape(blackAndWhiteImage)
            contours, hierarchy = cv2.findContours(blackAndWhiteImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.imwrite('output.jpg', blackAndWhiteImage)
	    
            locations = []
        for contour in contours:
            location = cv2.boundingRect(contour)
            x, y, w, h = location
            #print(w, h)
            if(w*h<30):
                continue
            locations.append(location)
            locations.sort(key=lambda x:x[0])
        #print(locations)

        symbol_segment_location = []
        # 每一個複合式字符合併
        symbol_segment_list = []
        extacted = []

        far_x = 0
        cur_x = 0
        min_y = 0
        max_y = 0
        idx = -1
        #print(len(locations))
        cur_idx = 0
        for i in range(len(locations)):
            '''if i == 0 or i == 1:
                continue'''
            #location = cv2.boundingRect(contour)
            x, y, w, h = locations[cur_idx]
            if w * h > size[0] * size[1] * 0.8 or w * h < size[0] * size[1] * (1/600.):
                del locations[cur_idx]
                continue
	  
            if idx == -1:
                idx = cur_idx
                cur_x = x
                far_x = x + w
                min_y = y
                max_y = y + h
            elif x >= cur_x - 1 and x + w <= far_x + 1:
                if y < min_y:
                    min_y = y
                    locations[idx] = tuple([cur_x, y, far_x - cur_x, max_y - min_y])
                if y + h > max_y:
                    max_y = y + h
                    locations[idx] = tuple([cur_x, min_y, far_x - cur_x, y + h - min_y])
                del locations[cur_idx]
                cur_idx -= 1
            elif x + w >= far_x and x < far_x:
                cur_x = x if x < cur_x else cur_x
                far_x = x + w
                if y < min_y:
                    min_y = y
                    locations[idx] = tuple([cur_x, y, far_x - cur_x, max_y - min_y])
                if y + h > max_y:
                    max_y = y + h
                    locations[idx] = tuple([cur_x, min_y, far_x - cur_x, y + h - min_y])
                del locations[cur_idx]
                cur_idx -= 1
            elif abs(x - far_x) < size[0] * 0.065:
                far_x = x + w
                min_y = y if y < min_y else min_y
                max_y = y + h if y + h > max_y else max_y
                locations[idx] = tuple([cur_x, min_y, far_x - cur_x, max_y - min_y])
                del locations[cur_idx]
                cur_idx -= 1
            elif x > far_x:
                idx = cur_idx
                cur_x = x
                far_x = x + w
                min_y = y
                max_y = y + h

            cur_idx += 1
        #print(locations)
        
        string = '1'
        for location in locations:
            symbol_segment_location.append(location)
            # 只提取方框內
            #extracted_img = extract_img(location,erosion,contour)
            extracted_img = extract_imgs(location, blackAndWhiteImage)
            symbol_segment_list.append(extracted_img)

            extacted.append(extracted_img)
            cv2.imwrite(string + 'output.jpg', extracted_img)
            string += '1'

        symbols=[]
        for i in range(len(symbol_segment_location)):
            symbols.append({'location':symbol_segment_location[i],'src_img':symbol_segment_list[i]})
            # 按照x座標由小到大排序

        symbols.sort(key=lambda x:x['location'][0])
        equa = ''
        pred_equa = ''
        for i in range(len(symbols)):
            src_img = np.expand_dims(symbols[i]['src_img'], axis=2)
            src_img = np.expand_dims(src_img, axis=0)
            pred = self.model.predict(src_img)
            pred = np.argmax(pred)
            label_sym = labelencoder[pred]
            print("!!!!!!!!!!")
            #print(pred)
            print(label_sym)
            print("!!!!!!!!!!")

            tmp = label_sym
            if label_sym == 'div':
                label_sym = '/'
                tmp = '÷'
            elif label_sym == 'times':
                label_sym = '*'
                tmp = 'x'
            elif label_sym == 'sin':
                label_sym = 'math.sin'
                tmp = 'sin'
            elif label_sym == 'cos':
                label_sym = 'math.cos'
                tmp = 'cos'
            elif label_sym == 'log':
                label_sym = 'math.log10'
                tmp = 'log'
            elif label_sym == 'pi':
                label_sym = 'math.pi'
                tmp = '𝝅'
            elif label_sym == '!':
                num = equa[len(equa) - 1]
                label_sym = 'math.factorial(' + num + ')'
                equa = equa[:len(equa) - 1]
                tmp = '!'
            equa += label_sym
            pred_equa += tmp
        return pred_equa, equa