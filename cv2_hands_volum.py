import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import math

import mediapipe as mp

Wcam, Hcam = 700, 400

cap = cv2.VideoCapture(0)
cap.set(3, Wcam)
cap.set(4, Hcam)

success, img = cap.read()
#imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
pTime = 0

mpDraw = mp.solutions.drawing_utils #desenha os pontos na imagem
mpHands = mp.solutions.hands
hands = mpHands.Hands()

def calcula_modulo(vet):
    comp = math.sqrt(np.abs(vet.x ** 2) + np.abs(vet.y ** 2))
    return comp
def calcula_vet_unitario(vet):
    mod = calcula_modulo(vet)
    Ux, Uy = vet.x/mod, vet.y/mod
    vet.x = Ux
    vet.y = Uy
    return vet

def calcula_angolo_do_vetor(V):
    U = calcula_vet_unitario(V)
    ang = math.acos(U.x)
    if U.y < 0:
        ang = -ang

    return ang

class Vetor:
    baseX = 0
    baseY = -100
    ang = 0.0
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def calcula_angulo(self):
        try:
            ang = calcula_angolo_do_vetor(self)
            self.ang = ang
        except Exception:
            pass
        
    def update_vetor_com_angulo(self):
        try:
            self.y = int(math.sin(self.ang)*100)
            self.x = int(math.cos(self.ang)*100)
        except Exception: 
            pass


dedo8 = Vetor(0,0)
dedo4 = Vetor(0,0)

def cria_vetor(d1, d2):
    xt = d1.x - d2.x
    yt = d1.y - d2.y
    x = Vetor(xt, yt)
    return x
entreDedos= cria_vetor(dedo8, dedo4)

botao = Vetor(0, 0)

def atualiza_entreDedos(d1, d2):
    xt = d1.x - d2.x
    yt = d1.y - d2.y
    entreDedos.x = xt
    entreDedos.y = yt



while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) #inverte a imagem na horizontal pra virar um "espelho"

    cTime = time.time()
    fps = 1/(cTime-pTime) #calculo de fps
    pTime = cTime

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #manipula a cor
    cv2.putText(img, f'FPS: {(int(fps))}', (380, 70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3) #mostra o fps no video

    ANGULO_BASE = entreDedos.ang
    result = hands.process(imgRGB) #extrai os pontos dos frames
    #print(result.multi_hand_landmarks)
    


    if result.multi_hand_landmarks: #se os pontos existirem
        for handLMK in result.multi_hand_landmarks: #separa eles
        
            for id, lm in enumerate(handLMK.landmark):
                #print(id, lm)
                h, w, _ = img.shape
                meioH = int(100)
                meioW = int(100)
                if id == 8:
                    dedo8.x, dedo8.y = int(lm.x*w), int(lm.y*h)
                    #cv2.circle(img, (cx8,cy8), 10, (255,0,255), cv2.FILLED)
                if id == 4:
                    dedo4.x, dedo4.y = int(lm.x*w), int(lm.y*h)
                    #cv2.circle(img, (cx4,cy4), 10, (255,0,255), cv2.FILLED)
                if dedo8 and dedo4:
                    atualiza_entreDedos(dedo8, dedo4)

                    comprimento = calcula_modulo(entreDedos)
                    color = (0,0,255)
                    
                    if comprimento < 90:
                        cv2.circle(img, (dedo8.x, dedo8.y), 15, (255,0,255), cv2.FILLED)
                        cv2.circle(img, (dedo4.x, dedo4.y), 15, (255,0,255), cv2.FILLED)
                        color = (0,255,255)
                        
            cv2.arrowedLine(img,(meioW, meioH), (botao.x + meioW, botao.y + meioH), (10,255,10), 5)
                    
            cv2.arrowedLine(img,(meioW+100, meioH), (entreDedos.x + meioW+100, entreDedos.y + meioH), color, 2)
            
            entreDedos.calcula_angulo()
            #print('entre',entreDedos.ang)
            if comprimento < 90:
                ANGULO_BASE = ANGULO_BASE - entreDedos.ang
                botao.ang = botao.ang - ANGULO_BASE
                botao.update_vetor_com_angulo()

                #print('botao',botao.ang)
            
            
                    

            mpDraw.draw_landmarks(img, handLMK, mpHands.HAND_CONNECTIONS) #e plota um a um no frame
    

    cv2.imshow("Image", img)
    cv2.waitKey(1)