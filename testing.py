import cv2
import numpy as np
import time
import face_recognition

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    
print(cap.isOpened())
process_this_frame = True
while True:
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # if len(face_encodings) == 0:
        #     frames_without_face += 1
        #     frames_with_face = 0
        #     if frames_without_face > 100:
        #         frames_without_face = 0
        #         await p.turn_off()
        # else:
        #     frames_with_face += 1
        #     frames_without_face = 0
        #     if frames_with_face > 10:
        #         frames_with_face = 0
        #         await p.turn_on()

        face_names = []
        for face_encoding in face_encodings:
            # TODO: add my face as known face
            # TODO: occlusion of face destroys recognition
            # TODO: add timings i.e. only turns on after 6 PM
            # TODO: turn off camera between 10 PM and 6 AM

            # See if the face is a match for the known face(s)
            # matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imwrite('Video.png', frame)
    # time.sleep(2)
    # if cv2.waitKey(1) == ord('q'):
    #     break


cap.release()