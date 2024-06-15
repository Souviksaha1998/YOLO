import cv2
import numpy as np
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

classList = ["car","fire"]
cell_Prediction  = (1 * 5) + len(classList)
rows = 7
cols = 7
# output_shape = np.zeros((rows,cols,cell_Prediction))
cell_H = int(448 // 7) 
cell_W = int(448 // 7)


class DataCreation(Dataset):
    def __init__(self,image_dir,txt_dir,transform=None):
        self.images_path = image_dir
        self.txt_path = txt_dir
        self.list_images = os.listdir(self.images_path)
        self.list_txt = os.listdir(self.txt_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, index):
    
        output_shape = np.zeros((rows,cols,cell_Prediction))
        image = cv2.imread(os.path.join(self.images_path,self.list_images[index]))
        H,W,C = image.shape
        txt = f"{os.path.join(self.txt_path,self.list_txt[index])}"
        with open(txt,"r") as f:
            coor = f.readlines()
            for c in coor:
                c = c.split()
                cls , x , y , w , h = map(float,c)
                
                # getting actual image x,y,w,h in pixels
                object_X = int(x*W)
                object_Y = int(y*H)
                # object_W = int(w*448)
                # object_H = int(h*448)
                
                # finding which grid cell
                g_x = object_X // cell_W
                g_y = object_Y // cell_H
                
                # making grid value as pixel wrt to cell
                grid_coor_in_pixl_X = g_x * 64
                grid_coor_in_pixl_Y = g_y * 64
                
                ###### finding relative X,Y point wrt to grid cell and rounding it to 6 floating points values
                relative_X = round((object_X - grid_coor_in_pixl_X) / cell_W,6)
                realtive_y = round((object_Y - grid_coor_in_pixl_Y) / cell_H,6)
                
                # W , H  already normalized by labelImg
                norm_W =  w
                norm_H =  h

                if output_shape[g_x,g_y,4] == 0:
                    output_shape[g_x,g_y,0:5] = [relative_X,realtive_y,norm_W,norm_H,1.0] 

                # elif output_shape[g_x,g_y,9] == 0:
                #     output_shape[g_x,g_y,5:10] = [relative_X,realtive_y,norm_W,norm_H,1.0] 
            
                output_shape[g_x,g_y,5+int(cls)] = 1.0
            
            
         
        if self.transform:
            image = self.transform(image)
            output_shape = self.transform(output_shape)
         
        return image , output_shape
        
        
    
# if __name__ == "__main__":
        

#     train_transform_ = transforms.Compose([
#     transforms.ToTensor()])

#     train_data = DataCreation("z","z1",train_transform_)
#     # test_data = Data("/content/test_data","/content/test_data_mask",transform=test_transform_)


#     train_dataloader = DataLoader(train_data,batch_size=2,shuffle=False)

#     # test_dataloader = DataLoader(test_data,batch_size=8,shuffle=False)
        
#     image , coor = next(iter(train_dataloader))
#     coor = coor.permute(0,2,3,1)
#     print(coor.shape)
    

    
    
    

















#0 0.973214 0.338170 0.053571 0.220982
# 0 0.068080 0.492188 0.131696 0.354911
# 0 0.478795 0.933036 0.145089 0.133929
# 0 0.643973 0.652902 0.131696 0.127232


















# cv2.circle(image,(int(x*448),int(y*448)),8,(0,255,0),-1) 
 # cv2.circle(image,(int(g_x*64),int(g_y*64)),8,(255,255,0),-1) 



# print(output_shape)

    
   









# cv2.imshow("image",image)
# cv2.waitKey(0)
