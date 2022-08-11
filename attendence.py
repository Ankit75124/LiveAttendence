import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#this will read all the images from my folder and create a list of it , it will auto update if we add any picture in the folder.

path='Images'
images=[]
classnames=[]
myList=os.listdir(path)
print("List Found")
print(f"There are total {len(myList)} pictures  in the list")

#it will import the images one by one , we can also use load image from face recognition module as we have done in "basic.py" but here we have used "imread" from os module.

for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])  # this line is add all the name in list and "os.path.splitext part is used to remove extention from file , like earlier it was ankit.jpg and now it is ankit"
# print(classnames)

# to find the encodings
def findEncodings(images):
    encodeList=[] #created a list
    for img in images:  # for travering through the list
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   #converting image from "BGR to RGB"
        encode=face_recognition.face_encodings(img)[0]  #it will encode all the files
        encodeList.append(encode)  # adding each images to the list
    return encodeList
def markAttendence(name):
    with open('atten.csv','r+') as f:
        myDataList=f.readlines()
        print(myDataList)
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


    

encodeListknown=findEncodings(images)
print('Encoding complete')
print(f"{len(encodeListknown) } images encoded")

#to intialize the webcam
cap=cv2.VideoCapture(0)

# we are using loop because camera will take multiple image through out the process , so we need to encode every image
while True:
    success, img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)  #we will resize the image coming from webcam for speed , because bigger image will take more time complete 
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB) #converting webcam image from "BGR to RGB"

    #in webcam we can find multiple faces so we have to find the location of all the faces and then compare it
    faceIncurFrame=face_recognition.face_locations(imgS)


    encodeCurFrame=face_recognition.face_encodings(imgS,faceIncurFrame)  #it will encode the image coming from webcam

    #finding the matches
    # we will iterate with all the faces, we will find encode face and encode loc of the image  
    for encondFace, faceLoc in zip(encodeCurFrame,faceIncurFrame):
        matches=face_recognition.compare_faces(encodeListknown,encondFace) #it will compare the list of our knowfaces to the encoded faces
        faceDis=face_recognition.face_distance(encodeListknown,encondFace) #it will find the distance of all the images in the list
        # print(faceDis)
        matchIndex=np.argmin(faceDis)  # this will give min distance of all the people from the list


        #to print their name on the screen of webcam
        if matches[matchIndex]:
            name=classnames[matchIndex].upper()
            # print(name)  #it will print name on terminal
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4 #we are multiplying the axis by 4 because earlier we have scale down the webcam image to 0.25 and denoted it by imgS 
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) #it will put name of the image closet in the screen
            markAttendence(name)
    #to show the image capturing from the webcam
    cv2.imshow('webcam',img)
    cv2.waitKey(1)  





 