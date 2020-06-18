import cv2
import numpy as np
# loading yolo algorithm
# dnn is deep neural network 
# Loading  Yolo weights --> pre-trained model and the configuration file
net = cv2.dnn.readNet("./Yolo/yolov3_custom_last.weights", "./Yolo/yolov3_custom.cfg") 
classes = list()
f= open("./Yolo/obj.names", "r") # read .names file
for line in f.readlines(): # loop each line in file
    classes.append(line.strip())  # split removes extra spaces and append the rest data into the list
layer_names = net.getLayerNames() # get layes names
layer=list() # init layes list
for i in net.getUnconnectedOutLayers():
    # we are getting the output layer to get detection of the object
    layer.append(layer_names[i[0] - 1])
colors = np.random.uniform(0, 255, size=(len(classes), 3))
#provide video path in string or 0== internal webcam 2== external webcam
cap = cv2.VideoCapture(0) # read input from default. 0 is for webcam 1 is for video and 2 is for exteral webcam 
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
while(cap.isOpened()):
    # get video Frame by Frame
    ret, frame = cap.read()
    if ret == True:
        img=frame#set img to frame.
        height, width,_ = img.shape #this will get image height and width 
        # create input blob to detect image
        #0.00392 is the scale factor
        #416,416 is the size we are passing to the algorithm
        # it basically prepares the input image to run through the deep neural network.
        #openCV works with BGR format  
        #so blob has 3 channels BGR 
        #416 * 416 image
        blobObject = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) 
        net.setInput(blobObject) 
        outputs = net.forward(layer)#get next layer
        #this basically helps us to extract the bounding box 
        class_ids = list() #helps detect class label 
        confidences = list() #assurity level of what an object is lower the value means that the system couldn't detect it 0.5 is threshold value 
        boxes = list() #bounding box 
        #displaying information on the screen 
        for out in outputs:
            for detection in out: 
                scores = detection[5:]  #detection[5:] returns probability of each score 
                class_id = np.argmax(scores) #class id is associated with class list name which tells about the object
                confidence = scores[class_id] 
                if confidence > 0.5: #tresh >0.5 it goes from 0 to 1
                   #object detected
                   #get center 
                   #when object will be detected it will mark a O on it 
                    center_x = int(detection[0] * width) #returns center x 
                    center_y = int(detection[1] * height) #return center y
                    #get height and width 
                    w = int(detection[2] * width) #2 gives wudth
                    h = int(detection[3] * height) #3 returns height 
                    # get points for rectangle
                    x = int(center_x - w / 2) #top left x
                    y = int(center_y - h / 2) #top left y 
                    #combine makes top left object
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        dec = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)#number of dectected objects in frame
        for i in range(len(boxes)):
            if i in dec:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])#get label
                color = colors[i]#each classs has specific color defined above
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)#create rectangle
                cv2.putText(img, label, (10,50), cv2.FONT_ITALIC, 2, color, 3)
        cv2.imshow("HAND GESTURE", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()