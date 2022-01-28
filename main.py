import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime 

path = 'ImagesAttendance'
images=[] 
classNames=[]
mylist=os.listdir(path) 
# print(mylist)

for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg) # image list
    classNames.append(os.path.splitext(cl)[0]) 


def findEncodings(images):
    encodeList=[] 
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # find encodings:
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList

def markAttendance(name): 
    with open('attendance.csv','r+') as f:
        myDataList=f.readlines() 
        namelist=[] 
        for line in myDataList: 
            entry=line.split(',')
            namelist.append(entry[0])
        
        if name not in namelist: 
            now=datetime.now() 
            time=now.strftime('%H:%M:%S') 
            date=now.strftime('%d/%m/%y')
            f.writelines(f'\n{name},{time},{date}') 


encodeListKnown=findEncodings(images)
# print(len(encodeListKnown))
print('Encoding Complete...')

# find the matches between our encodings
cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read() 
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        facedis=face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(facedis) 
       
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            # print(name)
            # create rectangle
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED) 
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img) 
    key=cv2.waitKey(1)
   

