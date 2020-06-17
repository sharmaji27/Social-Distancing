# Social-Distancing  
This project is made to monitor if Social Distancing is being followed in a locality or not.  
I have used the YOLO architecture based on darknet to build this project.  
You must have CUDA enabled OPENCV to use it as it is otherwise just comment out lines 9 and 10 and you are good to go.  
You can also see a counter on the bottom left corner indicating the number of violations in the frame.  
One can also use it on a webcam just by tweaking a single change in line 14 --> cap = cv2.VideoCapture(0).  
