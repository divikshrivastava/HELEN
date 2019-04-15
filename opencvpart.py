import cv2
import imutils
import numpy as np
import math
import pickle
bg = None
def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight) #The function calculates the weighted sum of the input 'image' and the accumulator 'bg' so that 'bg' becomes a running average of a frame sequence


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)[1]
    (_, cnts, _) = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #when no contours detected
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(segmented,True)
        approx= cv2.approxPolyDP(segmented,epsilon,True)

        #make convex hull around hand
        hull = cv2.convexHull(segmented)

        #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(segmented)
        if areacnt<20:
           areacnt=15
        #find the percentage of area not covered by hand in convex hull
    
        arearatio=((areahull-areacnt)/areacnt)*100
        
            
        #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx,hull)

        l=0#defects intial
        try:
            #code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt= (100,180)
            
            
                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
                #distance between point and convex hull
                d=(2*ar)/a
            
                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d>30:
                    l += 1
                    cv2.circle(roi, far, 3, [255,0,0], -1)
            
                #draw lines around hand
                cv2.line(roi,start, end, [0,255,0], 2)
            l+=1
            return (thresholded, segmented, l, arearatio, areacnt)
        except AttributeError:
            return (thresholded, segmented, 1, arearatio, areacnt)
    
if __name__ == "__main__":
    #if lower value is set, running avg wil be performed on larger amount of previous frames
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0

    while(True):
        (grabbed, frame)=camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        #Numpy slicing to concentrate on ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        
        retent=''
        if num_frames < 30:         
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)    
            
            if hand is not None:
                (thresholded, segmented,l,arearatio,areacnt) = hand
                if l==1:
                   if areacnt<2000:
                      retent='Put hand in the box'
                   else:
                        if areacnt<9000:
                           retent='I need help'
                        else:
                           retent='Emergency!'
                elif l==2:
                    if areacnt<12000:
                       retent='Yes'
                    elif areacnt>12000:
                       retent='No'
                    
                elif l==3:
                     if areacnt<13000:
                        retent='Call the cops!'
                     else:
                         retent='Call Medic!'
                    
                elif l==4  :
                     retent='Navigation'
            
                elif l==5:
                     retent='Hello! I am HELEN'
            
                elif l==6:
                     retent='re-position'
            
                else :
                     retent='re-position'    
                
                
                cv2.putText(clone,retent, (25,400), cv2.FONT_HERSHEY_SIMPLEX, 2,(51,255,51) ,2)
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == 27:
            break
camera.release()
cv2.destroyAllWindows()
