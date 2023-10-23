import cv2
import numpy as np
import asyncio
from kasa.smartplug import SmartPlug

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()


async def main():
    
    p = SmartPlug("192.168.10.78")
    await p.update()
    print(p.alias)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
        
    frames_with_face = 0
    frames_without_face = 0
    process_this_frame = True
    
    # Only process every other frame of video to save time
    while True:
        # TODO: add my face as known face
        # TODO: occlusion of face destroys recognition
        # TODO: add timings i.e. only turns on after 6 PM
        # TODO: turn off camera between 10 PM and 6 AM
        if process_this_frame:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        # We will focus on the class_id for 'person' which is typically 0
                        if class_id == 0:
                            # print("Human detected")

                            # Get bounding box coordinates
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                            # Draw rectangle
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            if len(indexes) == 0:
                frames_without_face += 1
                frames_with_face = 0
                if frames_without_face > 100:
                    frames_without_face = 0
                    await p.turn_off()
            else:
                frames_with_face += 1
                frames_without_face = 0
                if frames_with_face > 10:
                    frames_with_face = 0
                    await p.turn_on()

            # Draw bounding box
            # for i in range(len(boxes)):
            #     if i in indexes:
            #         x, y, w, h = boxes[i]
            #         label = str("human")
            #         confidence = confidences[i]
            #         color = (0, 255, 0)
            #         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            #         cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                 
        process_this_frame = not process_this_frame

        # Display the resulting image
        # cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())