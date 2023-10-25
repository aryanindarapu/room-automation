import utils
import asyncio
import requests
import json
import cv2
from kasa.smartplug import SmartPlug

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # reduce frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # reduce frame height


async def main():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    p = SmartPlug("192.168.10.78")
    await p.update()
    print(p.alias)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    frame_count = 0
    frames_with_motion = 0
    frames_without_motion = 0

    while True:
        if frame_count % 2 == 0 :  # Skip every other frame
            contours = utils.frame_diff(frame1, frame2)
            
            # if len(contours) > 30: # TODO: change this threshold 
            #     # TODO: call lambda api here and parse response
            #     print("Motion detected")
            #     frames_with_motion += 1
                
            if len(contours) < 5:
                frames_without_motion += 1
                frames_with_motion = 0
                # print("Frames without motion: " + str(frames_without_motion))
                if frames_without_motion > 100:
                    frames_without_motion = 0
                    await p.turn_off()
            else:
                frames_with_motion += 1
                frames_without_motion = 0
                # print("Frames with motion: " + str(frames_with_motion))
                if frames_with_motion > 30:
                    # print("Turning on")
                    frames_with_motion = 0
                    await p.turn_on()
                
            # for contour in contours:
            #     (x, y, w, h) = cv2.boundingRect(contour)
            #     if cv2.contourArea(contour) < 500:  # Ignore small movements
            #         continue
            #     cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # cv2.imshow("Motion Detector", frame1)

        frame1 = frame2
        ret, frame2 = cap.read()
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())