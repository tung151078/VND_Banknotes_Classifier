import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py
import cv2

menu = ['Welcome','Capture images by Webcam', 'Predict by Image file', 'About me']
choice = st.sidebar.selectbox('What can you do?', menu)


def main():
        if choice=='Welcome':
            st.title("VND Banknotes Classifier")
            st.write("This application is used to classify 9 currencies of Vietnam: 1000 VND, 2000 VND, 5000 VND, 10000 VND, 20000 VND, 50000 VND, 100000 VND, 200000 VND, 500000 VND ")
            st.write("")
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
            col7, col8, col9 = st.columns(3)

            # 1000 dong
            with col1:
                st.image("media/1000.jpg")
            # 2000 dong
            with col2:
                st.image("media/2000.jpg")
            # 5000 dong
            with col3:
                st.image("media/5000.jpg")
                # 1000 dong
            with col4:
                st.image("media/10000.jpg")
            # 2000 dong
            with col5:
                st.image("media/20000.jpg")
            # 5000 dong
            with col6:
                st.image("media/50000.jpg")
                # 1000 dong
            with col7:
                st.image("media/10000.jpg")
            # 2000 dong
            with col8:
                st.image("media/200000.jpg")
            # 5000 dong
            with col9:
                st.image("media/500000.jpg")
                
        elif choice=='Predict by Image file':
            file_uploaded = st.file_uploader("Please upload your file", type = ['jpg', 'png', 'jpeg'])
            if file_uploaded is not None:
                image = Image.open(file_uploaded)
                figure = plt.figure()
                plt.axis('off')
                result = predict_class(image)
                st.write(result)
                st.image(image)
                                    
        elif choice=='Capture images by Webcam':
            st.title('Open your webcam')
            st.warning('Press "c" to capture image & "q" to quit!')
            cap = cv2.VideoCapture(0)
            i=0 #to save all the clicked images
            while(True):
                ret, frame = cap.read()
                
                cv2.imshow("imshow",frame)
                key=cv2.waitKey(30)
                if key==ord('c'):
                    i+=1
                    cv2.imshow("imshow2",frame)
                    cv2.imwrite('C:/Users/tungt/CoderSchool/project/image/'+str(i)+'.jpg', frame)
                    print("Wrote Image")
                if key==ord('q'):
                    break
            # release the capture
            cap.release()
            cv2.destroyAllWindows()
        elif choice=='About me':
            st.balloons()
            st.title('Vu Thanh Tung')
            st.header('CoderSchool - ML30')
            
    
def predict_class(image):
  classifier_model = tf.keras.models.load_model(r'my_model.h5')
  shape = ((224,224,3))
  model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])
  test_image = image.resize((224,224))
  test_image = preprocessing.image.img_to_array(test_image)
  test_image = test_image/255.0
  test_image = np.expand_dims(test_image, axis=0)
  class_names = ['1000', '2000', '5000', '10000', '20000', '50000', '100000', '200000', '500000']
  predictions = model.predict(test_image)
  scores = tf.nn.softmax(predictions[0])
  scores = scores.numpy()
  image_class = class_names[np.argmax(scores)]
  result = 'The image uploaded is: {}'.format(image_class)
  return result

if __name__ == "__main__":
  main()