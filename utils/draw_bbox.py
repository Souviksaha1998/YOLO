import numpy as np
import cv2
import os
import random

def draw_bbox(image_path,show_grids=False):
    classList = ["car","fire"]
    rows = 7 
    cols = 7
    image = cv2.imread(image_path)
    H,W,C = image.shape
    txt = f"{os.path.splitext(image_path)[0]}.txt"
    with open(txt,"r") as f:
        coor = f.readlines()
        for c in coor:
            c = c.split()
            cls , x , y , w , h = map(float,c)
            x1 = int((x - (w/2))*W)
            y1 = int((y - (h/2))*H)
            x2 = int((x + (w/2))*W)
            y2 = int((y + (h/2))*H)
            r , g , b = random.randint(1,255) , random.randint(1,255) , random.randint(1,255)
            cv2.rectangle(image,(x1,y1),(x2,y2),(b,g,r),2)
            cv2.putText(image,f"{classList[int(cls)]}",(x1,y1-20),cv2.FONT_HERSHEY_COMPLEX,1,(b,g,r),2)
            
    if show_grids:
        for i in range(cols):
            cv2.line(image,(64*(i),0),(64*(i),448),(0,0,255),2,cv2.LINE_AA)
            if i == 6:
                cv2.line(image,(64*(i+1),0),(64*(i+1),448),(0,0,255),2,cv2.LINE_AA)

        for i in range(rows):
            cv2.line(image,(0,64*i),(448,64*i),(0,0,255),2,cv2.LINE_AA)
            if i == 6:
                cv2.line(image,(0,64*(i+1)),(448,64*(i+1)),(0,0,255),2,cv2.LINE_AA)

    cv2.imshow("image",image)
    cv2.waitKey(0)


# def preprocess(image_path):
#     output_shape = np.zeros((rows,cols,cell_Prediction))
 
#     image = cv2.imread(image_path)
#     H,W,C = image.shape
#     txt = f"{os.path.splitext(image_path)[0]}.txt"
#     with open(txt,"r") as f:
#         coor = f.readlines()
#         for c in coor:
#             c = c.split()
#             cls , x , y , w , h = map(float,c)
            
#             # getting actual image x,y,w,h in pixels
#             object_X = int(x*W)
#             object_Y = int(y*H)
#             # object_W = int(w*448)
#             # object_H = int(h*448)
            
#             # finding which grid cell
#             g_x = object_X // cell_W
#             g_y = object_Y // cell_H
            
#             # making grid value as pixel wrt to cell
#             grid_coor_in_pixl_X = g_x * 64
#             grid_coor_in_pixl_Y = g_y * 64
            
#             ###### finding relative X,Y point wrt to grid cell
#             relative_X = round((object_X - grid_coor_in_pixl_X) / cell_W,6)
#             realtive_y = round((object_Y - grid_coor_in_pixl_Y) / cell_H,6)
            
#             # W , H  already normalized by labelImg
#             norm_W =  w
#             norm_H =  h

#             if output_shape[g_x,g_y,4] == 0:
#                 output_shape[g_x,g_y,0:5] = [relative_X,realtive_y,norm_W,norm_H,1.0] 

#             elif output_shape[g_x,g_y,9] == 0:
#                 output_shape[g_x,g_y,5:10] = [relative_X,realtive_y,norm_W,norm_H,1.0] 
        
#             output_shape[g_x,g_y,10+int(cls)] = 1.0

#     return image , output_shape