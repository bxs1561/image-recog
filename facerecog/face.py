import cv2
import time
import pandas
from datetime import datetime


def image_recog():

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread("school.jpg")
    img = cv2.resize(img, (600, 600))


    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)
    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y), (x+w, y+h), (255, 0, 0), 3)

    cv2.imshow("School", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# cap.release()
def image():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)
    # To use a video file as input
    # cap = cv2.VideoCapture('filename.mp4')

    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display
        cv2.imshow('Video', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    # Release the VideoCapture object
    cap.release()

def image_detect():
    img = cv2.imread("obama.jpg")
    resize = cv2.resize(img, (600, 600))

    # other way to resize image
    # resize = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

    cv2.imshow("Obama", resize)
    # wait for some time if put number bigger than 0
    # 0 indicates enter to cancel image view
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def face_detect():
    # Create cascade classifier object
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Reading image
    img = cv2.imread("obama.jpg")

    # Read image as gray color scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Search the coordinates of the image
    # scaleFactor : Decreases the shape of the value by 5 percent until the faces found, smaller value is greater accuracy
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("Obama", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def motion_detector():
    first_frame = None
    status_list = [None, None]
    times=[]

    # store the time values during which object detection and movement appears
    df = pandas.DataFrame(columns=["Start", "End"])

    # Create a video capture object to record video
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        # object is not visible at the beginning, status is 0
        status =0

        # convert frame color to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame to GaussianBlur
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Store the first image to image of the video
        # whatever image come in the camera when it is on
        if first_frame is None:
            first_frame = gray
            continue
        # change status when object is being detect
        status +=1

        # Calculate difference between the first and other frames
        delta_frame = cv2.absdiff(first_frame, gray)

        # if the difference is greater than 30, it will convert the pixels to white
        thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)

        # Define the contours area. Basically add the borders
        cnts, hierarchy = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remove noises and shadows. Basically, it will keep only that part white , which has area greater than 1000 pixels
        for contour in cnts:
            if cv2.contourArea(contour) < 1000:
                continue

            (x, w, y, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # list of status for every frame
        status_list.append(status)

        status_list = status_list[-2:]

        # Record datetime in a list when change occurs
        if status_list[-1] == 1 and status_list[-2] == 0:
            times.append(datetime.now())

        if status_list[-1] == 0 and status_list[-2] == 1:
            times.append(datetime.now())

        # Store times in dataframe
        for i in range(0, len(times), 2):
            df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)

        df.to_csv("Times.csv")

        cv2.imshow("frame", frame)
        cv2.imshow("Capturing", gray)
        cv2.imshow("delta", delta_frame)
        cv2.imshow("thresh", thresh_delta)
    video.release()
    cv2.destroyAllWindows()


def main():
    image_recog()
    # motion_detector()
    # face_detect()
    # image_detect()

if __name__ == '__main__':
    main()
