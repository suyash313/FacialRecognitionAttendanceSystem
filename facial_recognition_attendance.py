# Static Facial Recognition
"""
import cv2
import face_recognition

aryan = face_recognition.load_image_file('photos for training/aryan.jpeg')
aryan = cv2.cvtColor(aryan, cv2.COLOR_BGR2RGB)
atest = face_recognition.load_image_file('photos for testing/atest.jpeg')
atest = cv2.cvtColor(atest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(aryan)[0]
encodeimg = face_recognition.face_encodings(aryan)[0]
cv2.rectangle(aryan, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

faceloctest = face_recognition.face_locations(atest)[0]
encodetest = face_recognition.face_encodings(atest)[0]
cv2.rectangle(atest, (faceloctest[3], faceloctest[0]), (faceloctest[1], faceloctest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeimg], encodetest)
facedist = face_recognition.face_distance([encodeimg], encodetest)
print(results, facedist)
cv2.putText(atest, f'{results} {round(facedist[0], 2)}', (60, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

cv2.imshow('aryan', aryan)
cv2.imshow('testaryan', atest)
cv2.waitKey(0)
"""

# Real Time Facial Recognition Using Webcam


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

path = 'photos for training'  # Path for the photos to train the model.
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

        cv2.imshow('Live Feed', img)
        cv2.waitKey(1)



