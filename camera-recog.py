import cv2
import numpy as np
import face_recognition
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
        
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    frames_without_face = 0
    frames_with_face = 0
    while True:
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
                        print("Human detected")

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

        # Draw bounding box
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str("human")
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        # Only process every other frame of video to save time
        # if process_this_frame:
        #     # Resize frame of video to 1/4 size for faster face recognition processing
        #     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        #     # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        #     rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
            
        #     # Find all the faces and face encodings in the current frame of video
        #     face_locations = face_recognition.face_locations(rgb_small_frame)
        #     face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        #     if len(face_encodings) == 0:
        #         frames_without_face += 1
        #         frames_with_face = 0
        #         if frames_without_face > 100:
        #             frames_without_face = 0
        #             await p.turn_off()
        #     else:
        #         frames_with_face += 1
        #         frames_without_face = 0
        #         if frames_with_face > 10:
        #             frames_with_face = 0
        #             await p.turn_on()
            
        #     face_names = []
        #     for face_encoding in face_encodings:
        #         # TODO: add my face as known face
        #         # TODO: occlusion of face destroys recognition
        #         # TODO: add timings i.e. only turns on after 6 PM
        #         # TODO: turn off camera between 10 PM and 6 AM
                
        #         # See if the face is a match for the known face(s)
        #         # matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        #         name = "Unknown"

        #         # # If a match was found in known_face_encodings, just use the first one.
        #         # if True in matches:
        #         #     first_match_index = matches.index(True)
        #         #     name = known_face_names[first_match_index]

        #         # Or instead, use the known face with the smallest distance to the new face
        #         # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #         # best_match_index = np.argmin(face_distances)
        #         # if matches[best_match_index]:
        #         #     name = known_face_names[best_match_index]

        #         face_names.append(name)

        # process_this_frame = not process_this_frame
        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #     top *= 4
        #     right *= 4
        #     bottom *= 4
        #     left *= 4

        #     # Draw a box around the face
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #     # Draw a label with a name below the face
        #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())