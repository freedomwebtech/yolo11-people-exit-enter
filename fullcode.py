import cv2
import numpy as np
from ultralytics import YOLO
import cvzone




# Mouse callback function for RGB window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load YOLO11 model
model = YOLO("yolo11n.pt")
names = model.names



# Open the video file or webcam
cap = cv2.VideoCapture('peoplecount1.mp4')
count=0
area1=[(250,444),(211,448),(473,575),(514,566)]
area2=[(201,449),(177,453),(420,581),(457,577)]
enter={}
list=[]
exit={}
list1=[]
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 600))
    
    # Run YOLO11 tracking on the frame
    results = model.track(frame, persist=True)
    
  
    
    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes, class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores
        
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            if 'person' in c:
                x1, y1, x2, y2 = box
                result=cv2.pointPolygonTest(np.array(area2,np.int32),((x1,y2)),False)
                if result>=0:
                   enter[track_id]=(x1,y2) 
                if track_id in enter:
                    result1=cv2.pointPolygonTest(np.array(area1,np.int32),((x1,y2)),False)
                    if result1>=0:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                        cvzone.putTextRect(frame,f'{track_id}',(x1,y1),1,1)
                        cv2.circle(frame,(x1,y2),4,(255,0,0),-1)
                        if list.count(track_id)==0:
                           list.append(track_id)
###########################################################################################
                result2=cv2.pointPolygonTest(np.array(area1,np.int32),((x1,y2)),False)
                if result2>=0:
                   exit[track_id]=(x1,y2) 
                if track_id in exit:
                    result3=cv2.pointPolygonTest(np.array(area2,np.int32),((x1,y2)),False)
                    if result3>=0:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                        cvzone.putTextRect(frame,f'{track_id}',(x1,y1),1,1)
                        cv2.circle(frame,(x1,y2),4,(255,0,0),-1)
                        if list1.count(track_id)==0:
                           list1.append(track_id)           
        
            
    
    enterinshop=len(list)
    exitfromshop=len(list1)
    cvzone.putTextRect(frame,f'{enterinshop}',(50,60),2,2)
    cvzone.putTextRect(frame,f'{exitfromshop}',(50,100),2,2)

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,255),2)

    # Display the frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
