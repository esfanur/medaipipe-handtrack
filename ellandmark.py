import cv2
import mediapipe.python.solutions.hands

#mediapipe resımlerı rgb formatında alıyor opencv ıse bgr o yuzden once medıapıpe ıcı donsum yapacagız

cam=cv2.VideoCapture(0)

mphands=mediapipe.solutions.hands #bu bızım landmarklar arasındakı baglantıları cızmemıze yarayacak

hands=mphands.Hands()#bir el objesı olusturcak bızım ıcın

mpDraw=mediapipe.solutions.drawing_utils#buda bıze elımızdekı noktaları alıp cızdrıcek
while 1:
    ret,frame=cam.read()

    frame=cv2.flip(frame,1)

    framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    #rgb formata cevırfdıgımız goruntunun ıcerısındekı noktaları bıze bulacak
    handland=hands.process(framergb)
    print(handland.multi_hand_landmarks) #noktalarımızı konsolda gormek ıcın

    if handland.multi_hand_landmarks:#none degılse yanı bızım degerkerımız kandmarklarımızı cızdırmek ıcun
        for landmarksnok in handland.multi_hand_landmarks: #21 elamanlı arrayımız oluyor 21 tane landmark noktamız var ve 21 elamanlı landmarklar arasınd connectıon bastıralım

           x,y=landmarksnok.landmark[0].x,landmarksnok.landmark[0].y
           x1,y1=landmarksnok.landmark[4].x,landmarksnok.landmark[4].y

           font=cv2.FONT_HERSHEY_COMPLEX
           if y1>y:
               cv2.putText(frame,"OLUMSUZ",(10,50),font,2,(255,0,0),2)
           else:
               cv2.putText(frame,"OLUMLU",(10,50),font,3,(0,0,0),2)



           mpDraw.draw_landmarks(frame,landmarksnok,mphands.HAND_CONNECTIONS)# burda ılk deger uzeıne cızecegımız kaynak ıkınıcısı landmark lıstesı  dıgerı ıse landmarklar arası baglantı onun ıcınde yukarda mphand olusturmustuk



    cv2.imshow("resim",frame)

    if cv2.waitKey(25) &0XFF==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()


