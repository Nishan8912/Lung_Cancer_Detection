import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from io import BytesIO
from PIL import Image

st.title("Lung Cancer Prediction Web App")

@st.cache(allow_output_mutation=True)
def load_model(model_name): 
  model = tf.keras.models.load_model(model_name)
  return (model)
model = load_model("C:\Vacation Practices\stremlit practice\data\CancerModels3")

class_names = ['Bengin case', 'Malignant case', 'Normal case']

def read_file_as_image(data)-> np.ndarray:
  image = np.array(Image.open(BytesIO(data)))
  return image

nav = st.sidebar.radio("Navigation",['Home', 'Prediction',"About Model"])

#--------------------------------------HOME----------------------------------------------------------------------------
if nav=='Home':
  text , images= st.columns(2)
  images.markdown('#')
  images.image('data//national-cancer-institute-59pGROkKJPE-unsplash.jpg')
  images.video('https://www.youtube.com/watch?v=AIz-nN2bzLk')
  text.header("What is Lung Cancer?")
  text.write("Cancer is a disease in which cells in the body grow out of control. When cancer starts in the lungs, it is called lung cancer.Lung cancer begins in the lungs and may spread to lymph nodes or other organs in the body, such as the brain.")    
  text.write("Cancer from other organs also may spread to the lungs. When cancer cells spread from one organ to another, they are called metastases.Lung cancers usually are grouped into two main types called small cell and non-small cell (including adenocarcinoma and squamous cell carcinoma).")   
  text.write("These types of lung cancer grow differently and are treated differently. Non-small cell lung cancer is more common than small cell lung cancer.")
  text.write("Cigarette smoking is the number one cause of lung cancer. Lung cancer also can be caused by using other types of tobacco (such as pipes or cigars), breathing secondhand smoke, being exposed to substances such as asbestos or radon at home or work, and having a family history of lung cancer.")
  st.markdown("##")
  st.markdown("##")
  
  dis, photo = st.columns(2)
  dis.header("What Is a Tumor?")
  dis.write("A tumor is an abnormal mass or growth of tissue in the body that serves no specific purpose. It can develop when cells grow and divide too quickly.")
  dis.subheader("Benign (Noncancerous) Tumors")
  dis.write("A benign tumor is made up of cells that aren't a threat to invade other tissues, and the tumor cells are contained within the tumor. The cells generally aren't very different from the surrounding cells, and they aren't highly abnormal.")
  dis.markdown("""
              Usually, benign types of tumors are harmless unless they are:
              - Pressing on nearby tissues, nerves, or blood vessels
              - Taking up space in the brain
              - Causing damage
              - Causing excess hormone production
              """)
  dis.subheader("Malignant (Cancerous) Tumors")
  dis.write("Malignant means that the tumor is made of cancer cells that can grow uncontrollably and invade nearby tissues. The cancer cells in a malignant tumor tend to be abnormal, and very different from the normal surrounding tissue.Some cancer cells can travel through the bloodstream or lymph system to other parts of the body. This spreading process is called metastasis.")
  photo.markdown('##')
  photo.image("data//national-cancer-institute-p1zy6izFI0M-unsplash.jpg")
  photo.markdown('##')
  photo.image("data//istockphoto-1289369001-612x612.jpg")

#------------------------------------PREDICTION-------------------------------------------------------------------------
if nav=="Prediction":
  st.header('Welcome! You can test your chest CT scan here')
  first, secod = st.columns(2)
  file_uploaded = first.file_uploader('Upload CT scan image')
  if file_uploaded is not None:
    img = Image.open(file_uploaded)
    img = img.resize((512,512))
    img = img.convert('RGB')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, 0)
    prediction = model.predict(img_batch)
    class_names = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0]) *100
    confidence = round(confidence, 2)
    st.image(img, width=250)
    
  if class_names=='Bengin case':
    one, two = st.columns(2)
    one.write('The predicted case is:')
    one.write('Bengin Case')
    two.write(f"Predicted Confidence:")
    two.write(f"{confidence}%")
    
  if class_names=='Malignant case':
    one, two = st.columns(2)
    one.write('The predicted case is:')
    one.write('Malignant Case')
    two.write(f"Predicted Confidence:")
    two.write(f"{confidence}%")
    
  if class_names =='Normal case':
    one, two = st.columns(2)
    one.write('The predicted case is:')
    one.write('Normal Case')
    two.write(f"Predicted Confidence:")
    two.write(f"{confidence}%")
    

#----------------------------ABOUT_MODEL----------------------------------------------------------------------------------
if nav =="About Model":
  st.header('Trained Model Profile')
  st.write('Visual Representation of CNN architecture of trained model: ')
  st.image('data//visual CNN model.png')
  st.write("The trained model has following profile of Accuracy: ")
  st.image('data//download.png')
  st.write("The prediction on some test data set is shown below: ")
  st.image('data//test_prediction.jpg')
