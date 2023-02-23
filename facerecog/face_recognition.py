import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
people = ['Bibek Subedi', 'Bikram', 'Nikhil Upreti', 'Rajesh Hamal']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")


def video_face_detection():
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv.VideoCapture(0)
    while True:
        # Read the frame
        _, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detect face in the object
        faces_react = face_cascade.detectMultiScale(gray, 1.1, 4)
        for x, y, w, h in faces_react:
            faces_roi = gray[y:y + h, x:x + w]

            label, confidence = face_recognizer.predict(faces_roi)

            print(f"label = {people[label]} with a confident of {confidence}")
            cv.putText(img, str(people[label]), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        cv.imshow("Video", img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()



def image_face_detection():
    img = cv.imread(r"/Users/bikramsubedi/PycharmProjects/facerecog/images/Bibek Subedi/2.jpg")
    # img = cv.resize(img, (600, 600))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Person", gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # for x, y, w, h in faces:
    #     cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2 )
    for x, y, w, h in faces:
        faces_roi = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(faces_roi)
        cv.putText(img, str(people[label]), (200, 450), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), thickness=3)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv.imshow("Face Detection", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    # image_face_detection()
    video_face_detection()


if __name__ == '__main__':
    main()
