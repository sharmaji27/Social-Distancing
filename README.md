# Social-Distancing  
1. This project is made to monitor if Social Distancing is being followed in a locality or not.  
2. I have used the YOLO architecture based on darknet to build this project.  
3. You must have CUDA enabled OPENCV to use it as it is otherwise just comment out lines 9 and 10 and you are good to go.  
4. You can also see a counter on the bottom left corner indicating the number of violations in the frame.  
5. One can also use it on a webcam just by tweaking a single change in line 14 --> cap = cv2.VideoCapture(0).  

Download yolov3.weights file from here and paste it in the master folder with cfg file --> https://drive.google.com/u/0/uc?export=download&confirm=phEB&id=1I92QB0ifKoEOZcUq6gAfay-o4Tjw1xTU  

![](data/SD.gif)
