#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load imports
from scipy.io import wavfile as wav
import numpy as np
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt


# In[2]:


import pandas as pd
filepath_or_buffer = '/home/ece/Videos/bike sp hj.csv'
pd.read_fwf(filepath_or_buffer, colspecs='infer', widths=None, infer_nrows=100)


# In[3]:


def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
     
    return mfccsscaled


# In[4]:


# Load various imports 
import pandas as pd
import os
import librosa

# Set the path to the full UrbanSound dataset 
fulldatasetpath = '/home/ece/Videos/bike sp'

metadata = pd.read_csv('/home/ece/Videos/bike sp hj.csv')

features = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),str(row["file_name"]))
    class_label = row["class_name"]
    
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files') 


# In[5]:


print(metadata.class_name.value_counts())


# In[7]:


import pickle
file1=open('/home/ece/Music/features FINAL','wb')
pickle.dump(featuresdf,file1)
file1.close()


# In[8]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 


# In[9]:


# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.30, random_state = 42)


# In[10]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

num_labels = yy.shape[1]
filter_size = 4

# Construct model 
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))



model.add(Dense(num_labels))
model.add(Activation('softmax'))


# In[11]:


# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 


# In[18]:


# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)


# In[22]:


from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 400
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='/home/ece/Music/weigh FINALfeatures.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[23]:


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


# In[24]:


y_pred =  model.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))  
print(confusion_matrix)


# In[25]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


cm =confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))  
index = ["KAWASAKI","KTM","PULSAR","APACHE","HARLEY","ENFIELD"]                   
columns = ["KAWASAKI","KTM","PULSAR","APACHE","HARLEY","ENFIELD"]  
cm_df = pd.DataFrame(cm,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, annot=True)


# In[26]:


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


# In[27]:



#global cur_labelb
#global print_prediction
#global cur_labelb
def print_prediction(file_name):
        prediction_feature = extract_feature(file_name) 
        predicted_vector = np.argmax(model.predict(prediction_feature), axis=-1)
        predicted_class = le.inverse_transform(predicted_vector) 
        print("The predicted class is:", predicted_class[0], '\n') 
        global cur_labelb
        cur_labelb=ttk.Label(canvas,text = str("The predicted class is:")+str( predicted_class[0]), style='sp.TLabel')
        cur_labelb.place(x=790,y=900)
        #label1 = tk.Label(root, text = str("The predicted class is:")+str( predicted_class[0])).place(x=600,y=900)
        #canvas.create_text(400,200, text = str("The predicted class is:")+str( predicted_class[0]),font =("Helvetica",15),fill="white")
        predicted_proba_vector = model.predict(prediction_feature) 
        predicted_proba = predicted_proba_vector[0]
               
        for i in range(len(predicted_proba)): 
            pro.append(format(predicted_proba[i], '.5f'))
            category = le.inverse_transform(np.array([i]))
            print(category[0], "\t\t : ", format(predicted_proba[i], '.5f') )
            cur_label='Label'+str(i)
            cur_label=ttk.Label(canvas,text = str(category[0])+" = "+str(format(predicted_proba[i], '.3f')),style='green/black.TLabel')
            cur_label.grid(column=100,row=i+300,sticky='')
            #label = tkinter.Label(canvas, text = str(category[0])+"\t\t : "+str(format(predicted_proba[i], '.5f'))).grid(x=i+100,y=i+200)
                 
        


# In[28]:



def record(): 
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "file2.wav"
 
    audio = pyaudio.PyAudio()
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print ("recording...")
    frames = []
 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")
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
    filename = '/home/ece/Music/file2.wav' 
    print_prediction(filename)

def restart():
    cur_labela=ttk.Label(canvas,text = "                        ",style='green/black.TLabel')
    cur_labela.place(x=790,y=850)
    cur_labelb=ttk.Label(canvas,text = "                                     ", style='sp.TLabel')
    cur_labelb.place(x=790,y=900)
   
  
def play():
    pygame.mixer.init()
    pygame.mixer.music.load("/home/ece/Music/file2.wav")
    pygame.mixer.music.play(loops=0)
    
def openfile():
    filepath=filedialog.askopenfilename()
    print(filepath)
    print_prediction(filepath)
    


# In[29]:


import pygame
import os
import pyaudio
import wave
from tkinter import ttk


# In[30]:


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
ttk.Style().configure('green/black.TButton',font=('Helvetica', 20,'bold'), foreground='#1164e8', background='#010101')
ttk.Style().configure('sp.TLabel',font=('ariel', 25,'bold'), foreground='#e25d12', background='#020613')

button_rec = ttk.Button(canvas, text='START' ,style='green/black.TButton',command=record)
button_rec.place(x=500,y=750)

button_rec = ttk.Button(canvas, text='PREDICT',style='green/black.TButton',command=prd)
button_rec.place(x=1280,y=750) 

button_rec = ttk.Button(canvas, text='PREDICT from FILE',style='green/black.TButton',command=openfile)
button_rec.place(x=1280,y=850) 

button_rec = ttk.Button(canvas, text='RESET',style='green/black.TButton',command=restart)
button_rec.place(x=880,y=950)

play_button = ttk.Button(canvas, text='PLAY',style='green/black.TButton', command=play)
play_button.place(x=880,y=280)
mainloop()


# In[ ]:




