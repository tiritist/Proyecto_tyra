import numpy as np
import face_recognition as fr
import cv2
import os
import os 
def get_encoded_faces ():
    """
    Busca en la carpeta de rostros y codifica todos los rostros 
    :return: dict de (nombre, imagen codificada)
    """
encoded = {}
for dirpath, dnames, fnames in os.walk("./faces"):
    for f in fnames:
        if f.endswith(".jpg") or f.endswith(".png"):
            face = fr.load_image_file("faces/" + f)
            encoding = fr.face_encodings(face)[0]
            encoded[f.split(".")[0]] = encoding
return encoded
