import cv2
import numpy as np

net = cv2.dnn.readNet("./backup/yolov3_training_last.weights",'./backup/yolov3_testing.cfg')

classes = []
with open("./backup/classes.txt", "r") as f:
    classes = f.read().splitlines()

# cap = cv2.VideoCapture('jetbot_Trim.mp4')
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))



# Got rid of while loop for the purposes of the Kalman filter. 
# Also, made it a function for the same reason.
# TO DO: get the while loop back
def record():
    x_measured = 0
    y_measured = 0

    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

                x_measured, y_measured = x, y
                # print(x, y)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

    # States the class and the accuracy of the detection.
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('Image', img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    #cap.release()
    cv2.destroyAllWindows()
    return x_measured, y_measured

######


# from threading import Thread
# import cv2, time
# import numpy as np

# class ThreadedCamera(object):

#     def __init__(self, src=0):
#         self.capture = cv2.VideoCapture(src)
#         self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

#         # FPS = 1/X
#         # X = desired FPS
#         self.FPS = 1/30
#         self.FPS_MS = int(self.FPS * 1000)

#         # Start frame retrieval thread
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()

#     def update(self):
#         net = cv2.dnn.readNet('/usr/local/home/mt3qb/Desktop/Jetbot/jetbot_tracking/backup/yolov3_training_last.weights', '/usr/local/home/mt3qb/Desktop/Jetbot/jetbot_tracking/backup/yolov3_testing.cfg')

#         classes = []
#         with open("/usr/local/home/mt3qb/Desktop/Jetbot/jetbot_tracking/backup/classes.txt", "r") as f:
#             classes = f.read().splitlines()

#         colors = np.random.uniform(0, 255, size=(100, 3))
#         font = cv2.FONT_HERSHEY_PLAIN
        
#         while True:
#             if self.capture.isOpened():
#                 (self.status, self.frame) = self.capture.read()
#                 height, width, _ = self.frame.shape

#                 blob = cv2.dnn.blobFromImage(self.frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
#                 net.setInput(blob)
#                 output_layers_names = net.getUnconnectedOutLayersNames()
#                 layerOutputs = net.forward(output_layers_names)

#                 boxes = []
#                 confidences = []
#                 class_ids = []

#                 for output in layerOutputs:
#                     for detection in output:
#                         scores = detection[5:]
#                         class_id = np.argmax(scores)
#                         confidence = scores[class_id]
#                         if confidence > 0.2:
#                             center_x = int(detection[0]*width)
#                             center_y = int(detection[1]*height)
#                             w = int(detection[2]*width)
#                             h = int(detection[3]*height)

#                             x = int(center_x - w/2)
#                             y = int(center_y - h/2)

#                             boxes.append([x, y, w, h])
#                             confidences.append((float(confidence)))
#                             class_ids.append(class_id)

#                 indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

#                 if len(indexes)>0:
#                     for i in indexes.flatten():
#                         x, y, w, h = boxes[i]
#                         label = str(classes[class_ids[i]])
#                         confidence = str(round(confidences[i],2))
#                         color = colors[i]
#                         cv2.rectangle(self.frame, (x,y), (x+w, y+h), color, 2)
#                         cv2.putText(self.frame, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
#             time.sleep(self.FPS)

#     def show_frame(self):
#         cv2.imshow('frame', self.frame)
#         cv2.waitKey(self.FPS_MS)

# if __name__ == '__main__':
#     threaded_camera = ThreadedCamera()
#     while True:
#         try:
#             threaded_camera.show_frame()
#         except AttributeError:
#             pass
