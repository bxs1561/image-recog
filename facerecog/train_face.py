import os
import cv2 as cv
import numpy as np

people = ['Bibek Subedi', 'Bikram', 'Nikhil Upreti', 'Rajesh Hamal']
DIR = r'/Users/bikramsubedi/PycharmProjects/facerecog/images'

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# p=[]
# for i in os.listdir("/Users/bikramsubedi/PycharmProjects/facerecog/images"):
#     p.append(i)
#
# print(p)

features=[]
labels=[]

def create_train():
    for person in people:

        path = os.path.join(DIR, person)
        label = people.index(person)



        # Go through each images of person from folder
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            #
            img_array = cv.imread(img_path)

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for x, y, w, h in faces:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
    return labels, features

def main():
    labels,features=create_train()
    print("Training Done........")
    features = np.array(features ,dtype="object")
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    # Train the recognizer on the features list and labels list
    face_recognizer.train(features, labels)
    print(f'Length of the features = {len(features)}')
    print(f'Length of the labels = {len(labels)}')
    face_recognizer.save("face_trained.yml")
    np.save("features.npy", features)
    np.save("labels.npy", labels)


if __name__ == '__main__':
    main()

# remove ds store
# find . -name ".DS_Store" -delete