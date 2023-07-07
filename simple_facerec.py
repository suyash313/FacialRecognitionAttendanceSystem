"""
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

path = 'photos for training'  # Path for the photos to train the model.
imageBackground = cv2.imread('background/frame.png')
training_photos = []  # Photos in the central Database
student_names = []  # To capture the names from the photos from the database.
l1 = os.listdir(path)

for i in l1:
    current_image = cv2.imread(f'{path}/{i}')  # Current image is read
    training_photos.append(current_image)  # and appended to this list
    student_names.append(os.path.splitext(i)[0])  # Names corresponding to the images are also appended to list


def encode_images(training_photos):  # function for the encoding of images
    encoded_images_list = []
    for img in training_photos:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Images are converted to RGB for better detection and recognition
        image_location = face_recognition.face_locations(img)[0]  # location of the face in the image
        # 0th index suggests that the first face that is detected in an image is encoded.
        image_encode = face_recognition.face_encodings(img)[0]  # images are encoded
        encoded_images_list.append(image_encode)  # and appended to this list
    return encoded_images_list


def attendance_marking_during_entry(name):  # function for marking attendance after face has been recognized
    with open('Attendance.csv', 'r+') as file:  # a csv file named Attendance records the attendance of a class
        # for a given lecture

        database_list = file.readlines()  # reads the names of students from the central database
        student_list = []  # names of students that will attend the class
        for line in database_list:
            entry = line.split(',')  # these two lines are to make sure that duplicate attendance is not recorded
            student_list.append(entry[0])  # if someone's present their attendance will be written in the Attendance.csv
        if name not in student_list:  # for students whose attendance are not already marked
            current_time = datetime.now()  # the moment their face is recognized their attendance is marked for that
            # particular timestamp
            current_time_string = current_time.strftime('%H:%M:%S')  # strftime is a inbuilt method to convert time
            # in string format
            todays_date = datetime.today()  # will store the attendance against the date on that day
            todays_date_string = todays_date.strftime('%d-%m-%Y')   # converts date in string format
            attendance = 'PRESENT'  # those who show up will get marked present.
            file.writelines(f'\n{name}, {current_time_string}, {todays_date_string}, {attendance}')  # attendance is
            # written in our Attendance.csv


known_encoded_images_list = encode_images(training_photos)  # encoded images from the function are stored in this list
print('Encoding Complete')

webcam_capture = cv2.VideoCapture(0)  # Live feed initialization
webcam_capture.set(3, 640)
webcam_capture.set(4, 480)

start_time = time.time()    # a timer to ensure that students coming late cannot be marked present
seconds = 100    # in real-life implementation this would be somewhere around 600 seconds or 10 minutes more than
# the scheduled time of class

while True:

    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > seconds:  # if timer runs out, i.e. the 10 minute mark passes, the attendance module will close
        break
    else:
        success, img = webcam_capture.read()  # Feed from Webcam is captured
        # (In this case I have used webcam, however in real-life implementation, we'd use a camera module which costs
        # around 500 rupees)
        reduced_image = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # The image from the live-feed is reduced in size
        # This is done in order to speed up the process of facial-detection and recognition. Puts less load on the
        # processors

        reduced_image = cv2.cvtColor(reduced_image, cv2.COLOR_BGR2RGB)  # reduced image converted to RGB

        current_frame_faces = face_recognition.face_locations(reduced_image)  # checks for the faces in current frame
        current_frame_encode = face_recognition.face_encodings(reduced_image, current_frame_faces)  # encodes the faces
        # in current frame

        for face_encode, face_location in zip(current_frame_encode, current_frame_faces):  # zip aggregates the
            # iterables and returns single iterator object
            matches = face_recognition.compare_faces(known_encoded_images_list, face_encode)  # we verify the faces in
            # the current frame with the faces we have in our database
            face_distance = face_recognition.face_distance(known_encoded_images_list, face_encode)  # we calculate the
            # distance i.e. the measure to which the faces match. Face-Recognition library uses a method in which it
            # takes in account 128 distinctive features of a human face that are computer generated measurements and
            # then calculates certain values corresponding to those features. It can then compare the two faces on the
            # basis of how much they match, and returns a value which can be achieved through face_distance method.

            matched_face_index = np.argmin(face_distance)  # finds the best match from our database. It'll be used to
            # return the name corresponding to the best match.

            if matches[matched_face_index]:
                name = student_names[matched_face_index].upper()  # returns the name of the best match
                y1, x2, y2, x1 = face_location  # for creating a rectangle around the face in current frame
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # since, we decreased the size of our image from the
                # feed earlier by 1/4th, we now have to increase the size of the coordinates by 4

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.rectangle(img, (x1, y2 - 25), (x2, y2), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, name + " | PRESENT", (x1 + 3, y2 - 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 0), 2)
                # face recognized and marked present.
                attendance_marking_during_entry(name)  # attendance marking function called after a face is recognized

        imageBackground[158:158+480, 52:52+640] = img
        # cv2.imshow('Live Feed', img)
        cv2.imshow('Background', imageBackground)
        cv2.waitKey(1)
"""

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

path = 'photos for training'  # Path for the photos to train the model.
imageBackground = cv2.imread('background/frame.png')
training_photos = []  # Photos in the central Database
student_names = []  # To capture the names from the photos from the database.
l1 = os.listdir(path)

for i in l1:
    current_image = cv2.imread(f'{path}/{i}')  # Current image is read
    training_photos.append(current_image)  # and appended to this list
    student_names.append(os.path.splitext(i)[0])  # Names corresponding to the images are also appended to list


def encode_images(training_photos):  # function for the encoding of images
    encoded_images_list = []
    for img in training_photos:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Images are converted to RGB for better detection and recognition
        image_location = face_recognition.face_locations(img)[0]  # location of the face in the image
        # 0th index suggests that the first face that is detected in an image is encoded.
        image_encode = face_recognition.face_encodings(img)[0]  # images are encoded
        encoded_images_list.append(image_encode)  # and appended to this list
    return encoded_images_list


def attendance_marking_during_entry(name):  # function for marking attendance after face has been recognized
    with open('Attendance.csv', 'r+') as file:  # a csv file named Attendance records the attendance of a class
        # for a given lecture

        database_list = file.readlines()  # reads the names of students from the central database
        student_list = []  # names of students that will attend the class
        for line in database_list:
            entry = line.split(',')  # these two lines are to make sure that duplicate attendance is not recorded
            student_list.append(entry[0])  # if someone's present their attendance will be written in the Attendance.csv
        if name not in student_list:  # for students whose attendance are not already marked
            current_time = datetime.now()  # the moment their face is recognized their attendance is marked for that
            # particular timestamp
            current_time_string = current_time.strftime('%H:%M:%S')  # strftime is a inbuilt method to convert time
            # in string format
            todays_date = datetime.today()  # will store the attendance against the date on that day
            todays_date_string = todays_date.strftime('%d-%m-%Y')  # converts date in string format
            attendance = 'PRESENT'  # those who show up will get marked present.
            file.writelines(f'\n{name}, {current_time_string}, {todays_date_string}, {attendance}')  # attendance is
            # written in our Attendance.csv


known_encoded_images_list = encode_images(training_photos)  # encoded images from the function are stored in this list
print('Encoding Complete')


# def blink_detection(webcam_capture):
#     face_detector = FaceMeshDetector(maxFaces=1)
#     while True:
#         success1, blink_img = webcam_capture.read()
#         blink_img, faces = face_detector.findFaceMesh(blink_img)
#
#         cv2.imshow("Image", blink_img)
#         cv2.waitKey(1)

# face_cascades = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
# eye_cascades = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
#
# var = True
#
# while var:
#     success1, blinking_img = webcam_capture.read()
#     gray = cv2.cvtColor(blinking_img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bilateralFilter(gray, 5, 1, 1)
#
#     faces = face_cascades.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
#     if (len(faces) > 0):
#         for (x, y, w, h) in faces:
#             blinking_img = cv2.rectangle(blinking_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#             roi_face = gray[y:y + h, x:x + w]
#             roi_face_clr = blinking_img[y:y + h, x:x + w]
#             eyes = eye_cascades.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
#
#             if (len(eyes) >= 2):
#                 pass
#             else:
#                 print("Blink detected. Attendance Marked")
#                 var = False
#                 break


webcam_capture = cv2.VideoCapture(0)  # Live feed initialization
webcam_capture.set(3, 640)
webcam_capture.set(4, 480)

start_time = time.time()  # a timer to ensure that students coming late cannot be marked present
seconds = 3000  # in real-life implementation this would be somewhere around 600 seconds or 10 minutes more than
# the scheduled time of class

blinkCounter = 0
tempCounter = 0
face_detector = FaceMeshDetector(maxFaces=1)
eyePoints = [22, 23, 24, 25, 26, 110, 130, 157, 158, 159, 160, 161, 173, 190, 243, 249, 255, 373, 374, 380, 381, 384, 385, 386, 387, 388, 390, 398]
ratio_list = []
# 22, 23, 24, 26, 110, 130, 157, 158, 159, 160, 161, 243
# 27, 28, 29, 30, 56, 144, 145, 153

while True:

    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > seconds:  # if timer runs out, i.e. the 10 minute mark passes, the attendance module will close
        break
    else:
        success, img = webcam_capture.read()  # Feed from Webcam is captured
        # img = cv2.resize(img, (1200, 1080))

        # success1, img = webcam_capture.read()
        img, faces = face_detector.findFaceMesh(img, draw=False)
        if faces:
            face = faces[0] # because we only have 1 face

            # used to find the eyePoints
            """
            for id1 in range(len(face)):
                pos_x = face[id1][0]
                pos_y = face[id1][1]
                cv2.putText(
                    img,
                    f"{id1}",
                    (pos_x, pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                )
            """

            rightUp = face[159]
            rightDown = face[23]
            rightLeft = face[130]
            rightRight = face[243]
            rightLengthVertical, _ = face_detector.findDistance(rightUp, rightDown)
            rightLengthHorizontal, _ = face_detector.findDistance(rightLeft, rightRight)

            leftUp = face[386]
            leftDown = face[253]
            leftLeft = face[398]
            leftRight = face[255]
            leftLengthVertical, _ = face_detector.findDistance(leftUp, leftDown)
            leftLengthHorizontal, _ = face_detector.findDistance(leftLeft, leftRight)


            dist_ratio = int((rightLengthVertical/rightLengthHorizontal)*100)

            ratio_list.append(dist_ratio)
            if len(ratio_list) > 3:
                ratio_list.pop(0)
            ratio_average = sum(ratio_list) / len(ratio_list)

            if ratio_average < 35 and tempCounter == 0:
                blinkCounter += 1
                tempCounter = 1
            if tempCounter != 0:
                tempCounter += 1
                if tempCounter > 10:
                    tempCounter = 0

            if blinkCounter < 3:
                for point in eyePoints:
                    cv2.circle(img, face[point], 3, (255, 0, 255), cv2.FILLED)

                cv2.line(img, rightUp, rightDown, (0, 255, 0), 2)
                cv2.line(img, rightRight, rightLeft, (0, 255, 0), 2)

                cv2.line(img, leftRight, leftLeft, (0, 255, 0), 2)
                cv2.line(img, leftUp, leftDown, (0, 255, 0), 2)

            cvzone.putTextRect(img, f'Blink Count :  {blinkCounter}', (50, 75), scale=2)

        else:
            cvzone.putTextRect(img, 'NO FACE FOUND', (50, 250), scale=4)
            blinkCounter = 0
            # print("No face Found")
            # continue

        # (In this case I have used webcam, however in real-life implementation, we'd use a camera module which costs
        # around 500 rupees)
        reduced_image = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # The image from the live-feed is reduced in size
        # This is done in order to speed up the process of facial-detection and recognition. Puts less load on the
        # processors

        reduced_image = cv2.cvtColor(reduced_image, cv2.COLOR_BGR2RGB)  # reduced image converted to RGB

        current_frame_faces = face_recognition.face_locations(reduced_image)  # checks for the faces in current frame
        current_frame_encode = face_recognition.face_encodings(reduced_image, current_frame_faces)  # encodes the faces
        # in current frame

        for face_encode, face_location in zip(current_frame_encode, current_frame_faces):  # zip aggregates the
            # iterables and returns single iterator object
            matches = face_recognition.compare_faces(known_encoded_images_list, face_encode)  # we verify the faces in
            # the current frame with the faces we have in our database
            face_distance = face_recognition.face_distance(known_encoded_images_list, face_encode)  # we calculate the
            # distance i.e. the measure to which the faces match. Face-Recognition library uses a method in which it
            # takes in account 128 distinctive features of a human face that are computer generated measurements and
            # then calculates certain values corresponding to those features. It can then compare the two faces on the
            # basis of how much they match, and returns a value which can be achieved through face_distance method.

            matched_face_index = np.argmin(face_distance)  # finds the best match from our database. It'll be used to
            # return the name corresponding to the best match.

            # if blinkCounter==2:

            if matches[matched_face_index] and blinkCounter >= 2:
                name = student_names[matched_face_index].upper()  # returns the name of the best match
                y1, x2, y2, x1 = face_location  # for creating a rectangle around the face in current frame
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # since, we decreased the size of our image from the
                # feed earlier by 1/4th, we now have to increase the size of the coordinates by 4

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.rectangle(img, (x1, y2 - 25), (x2, y2), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, name + " | PRESENT", (x1 + 3, y2 - 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 0),
                            2)


                # blink_detection(webcam_capture)

                # face recognized and marked present.
                attendance_marking_during_entry(name)  # attendance marking function called after a face is recognized

        imageBackground[158:158 + 480, 52:52 + 640] = img

            # cv2.imshow("Image", blink_img)
            # cv2.waitKey(1)

        # cv2.imshow('Live Feed', img)
        cv2.imshow('Background', imageBackground)
        cv2.waitKey(1)
