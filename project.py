import cv2
import streamlit as st

st.title("Face Detection Project")

clf=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img_frm = st.image([])
camera = cv2.VideoCapture(0)

b=st.button("stop camera")
while True:
    _, frame = camera.read()
    st.text(frame)
    faces=clf.detectMultiScale(frame,1.2,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img_frm.image(frame)
    if b:
        break

