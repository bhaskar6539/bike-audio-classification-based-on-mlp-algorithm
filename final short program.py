#!/usr/bin/env python
# coding: utf-8

# ## loading saved feature and saved model for classification ##

# In[1]:


import pickle
import pickle
file=open('/home/ece/Music/features  gh FINAL','rb')
featuresdf=pickle.load(file)
file.close()


# In[2]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 


# In[3]:


# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


# In[4]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 


# In[5]:


import tensorflow as tf
from tensorflow import keras


# In[6]:


model=keras.models.load_model('/home/ece/Music/weigh FINAtures.hdf5')
model.summary()


# In[7]:


loss,acc=model.evaluate(x_test,y_test)
print("accuray:{:5.2f}%".format(100*acc))


# In[8]:


import librosa 
import numpy as np 

def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    return np.array([mfccsscaled])


# In[50]:


threshold=0.60
def print_predictions(file_name):
        prediction_feature = extract_feature(file_name) 
        predicted_vector = np.argmax(model.predict(prediction_feature), axis=-1)
        predicted_class = le.inverse_transform(predicted_vector) 
        
        global cur_labelb
        cur_labelb=ttk.Label(canvas,text = str("The predicted class is:")+str( predicted_class[0]), style='sp.TLabel')
        cur_labelb.place(x=790,y=900)
        predicted_proba_vector = model.predict(prediction_feature) 
        global predicted_proba
        predicted_proba = predicted_proba_vector[0]   
        for i in range(len(predicted_proba)): 
            
            category = le.inverse_transform(np.array([i]))
            if predicted_class[0] == category[0]:
                if threshold > float(format(predicted_proba[i],'.5f')):
                    cur_labels=ttk.Label(canvas,
                                         text = str("THE PREDICTAED CLASS IS WITH LOW CONFIDENCE")+str(format(predicted_proba[i],'.5f'))
                                         ,style='green/black.TLabel')
                    cur_labels.place(x=720,y=950)
                
        def small():
            child=Toplevel(canvas)
            child.geometry("400x200") 
            child.title("PREDICTED RESULT")
            child.configure(background='black')
            for i in range(len(predicted_proba)): 
                category = le.inverse_transform(np.array([i]))
                print(category[0], "\t\t : ", format(predicted_proba[i], '.5f') ) 
                cur_label='Label'+str(i)
                cur_label=ttk.Label(child,
                                text = str(category[0])+" = "+str(format(predicted_proba[i], '.3f')),
                                style='green/black.TLabel')
                cur_label.pack(fill = BOTH, expand = True)
        small()    
        


# In[51]:



def record(): 
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "filep.wav"
 
    audio = pyaudio.PyAudio()
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    #print ("recording...")
    frames = []
 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    #print ("finished recording")
    global cur_labela
    cur_labela=ttk.Label(canvas,text = "finished recording",style='green/black.TLabel')
    cur_labela.place(x=790,y=850)
 
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
def prd():
    global filepath
    filepath = '/home/ece/Music/filep.wav' 
    print_predictions(filepath)
    small()
def restart():
    cur_labela=ttk.Label(canvas,text = "                        ",style='green/black.TLabel')
    cur_labela.place(x=790,y=850)
    cur_labelb=ttk.Label(canvas,text = "                                     ", style='sp.TLabel')
    cur_labelb.place(x=790,y=900)
    for i in range(len(predicted_proba)): 
            cur_label='Label'+str(i)
            cur_label=ttk.Label(canvas,text ="                       ",style='green/black.TLabel')
            cur_label.grid(column=100,row=i+300,sticky='')
   
  
def play():
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play(loops=0)
    
def play1():
    pygame.mixer.init()
    pygame.mixer.music.load(filepath1)
    pygame.mixer.music.play(loops=0)    
    
    
def openfile():
    global filepath1
    filepath1=filedialog.askopenfilename()
    print_predictions(filepath1)
    


# In[52]:


import pygame
import os
import pyaudio
import wave
from tkinter import ttk


# In[53]:


from tkinter.ttk import *
from tkinter import *
import tkinter
from tkinter import filedialog
canvas = Tk()
canvas.geometry('1920x1080')
canvas = Canvas(width=1920, height=1080, bg='black')

canvas.pack(expand=YES, fill=BOTH)

gif1 = PhotoImage(file='fn.png')

canvas.create_image(0, 0, image=gif1, anchor=NW)
style = ttk.Style()
ttk.Style().configure('green/black.TLabel',font=('Helvetica', 20,'bold'), foreground='#e9d10a', background='#020613')
ttk.Style().configure('green/black.TButton',font=('Arial Black', 20,'bold'), foreground='#1164e8', background='#010101')
ttk.Style().configure('sp.TLabel',font=('ariel', 25,'bold'), foreground='#e25d12', background='#020613')
ttk.Style().configure('spW.TLabel',font=('ariel', 35,'bold'), foreground='#e5ce66', background='#020613')

cur_labeld=ttk.Label(canvas,text = 'BIKE SOUND CLASSIFICATION ', style='spW.TLabel')
cur_labeld.place(x=600,y=30)

cur_labeld=ttk.Label(canvas,text = 'BASED ON MACHINE LEARNING', style='spW.TLabel')
cur_labeld.place(x=600,y=80)

button_rec = ttk.Button(canvas, text='START' ,style='green/black.TButton',command=record)
button_rec.place(x=500,y=750)

button_rec = ttk.Button(canvas, text='CLASSIFY RECORDED FILE',style='green/black.TButton',command=prd)
button_rec.place(x=1280,y=750) 

button_rec = ttk.Button(canvas, text='CLASSIFY FROM FILE',style='green/black.TButton',command=openfile)
button_rec.place(x=1280,y=800) 

button_rec = ttk.Button(canvas, text='RESET',style='green/black.TButton',command=restart)
button_rec.place(x=500,y=800)

play_button = ttk.Button(canvas, text='PLAY RECORDED FILE',style='green/black.TButton', command=play)
play_button.place(x=700,y=950)

play_button = ttk.Button(canvas, text='PLAY AUDIO FROM FILE',style='green/black.TButton', command=play1)
play_button.place(x=980,y=950)
mainloop()


# In[ ]:




