'''
Created on Feb 19, 2014

@author: Hao
'''
import numpy as np
import cv2

def testCapture():

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()



def drawShapes():
    # defining an image 2x2x3 array and type
    img = np.zeros((512,512,3),np.uint8)  
    # drawing functions that operates on the first input object directly
    cv2.line(img, (0,0), (511,511),(255,0,0), 5)
    cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
    cv2.circle(img,(447,63), 63, (0,0,255), -1)
    cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

    #write something on canvas
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.CV_AA)

    cv2.namedWindow("window",cv2.WINDOW_AUTOSIZE)
    cv2.imshow("window", img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
    
    
if __name__ == "__main__":
    #testCapture()
    #drawShapes()
    
    
    
    