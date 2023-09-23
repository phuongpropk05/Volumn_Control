from cv2 import cv2
import time
import numpy as np
import hand as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(1)
pTime = 0

detector = htm.handDetector(detectionCon=0.7) #detectionCon độ tin cậy phát hiện tay 70%


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel() #
volRange=volume.GetVolumeRange()
print(volume.GetVolumeRange())

print(volRange[0])
print(volRange[1])
minVol = volRange[0]
maxVol = volRange[1]



while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList)!= 0:
        # cần sử dụng 2 ngón trỏ và ngón cái (point 4, và 8)
        x1, y1= lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # vẽ 2 đường tròn trên 2 đầu ngón cái và ngón trỏ
        cv2.circle(frame, (x1, y1), 15, (255, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 15, (255, 0, 255), -1)
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),3)

        # vẽ đường tròn giữa 2 đường thằng nối ngón cái và ngón giữa
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), -1)

        #xác định độ dài đoạn thẳng nối từ ngón trái đến ngón trỏ
        length= math.hypot(x2-x1,y2-y1)

        vol = np.interp(length,[25,250],[minVol,maxVol])
        print(length,vol)
        volume.SetMasterVolumeLevel(vol, None)
        volBar = np.interp(length, [25, 250], [400, 150])
        vol_tyle = np.interp(length, [25, 250], [0, 100])

        if length<25 :
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)

        cv2.rectangle(frame,(50,150),(100,400),(0,255,0),3)
        cv2.rectangle(frame,(50,int(volBar)),(100,400),(0,255,0),-1)

        # show %vol lên màn hình
        cv2.putText(frame, f"{int(vol_tyle)} %", (50, 130), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (150, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Volumn Control", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()