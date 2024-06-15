import torch 
import torch.nn as nn




# print(f"Device : {device}")

class Yolov1(nn.Module):
    def __init__(self,in_c=3,num_class=20):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c,64,7,stride=2,padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 , 192 ,3,padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(192 , 128 ,1,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128 , 256 ,3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256 , 256 ,1,padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256 , 512 ,3,padding=0),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(512 , 256 ,1,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256 , 512 ,3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512 , 512 ,1,padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512 , 1024 ,3,padding=0),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(1024 , 512 ,1,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512 , 1024 ,3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024 , 1024 ,3,padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024 , 1024 ,3,padding=1,stride=2),
            nn.LeakyReLU(0.1),
            
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024 , 1024 ,3,padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024 , 1024 ,3,padding=1),
            nn.LeakyReLU(0.1)
            
        )
        self.fc1 = nn.Linear(7*7*1024,4096)
        self.lrelu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(4096,7*7*(num_class+(5*2))) #xywhc..xywhc...c1..c2..c20
        
        
    
    
    def forward(self,x):
        print(f"input shape : {x.shape}")
        x = self.conv1(x)
        # print(f"conv1 : {x.shape}")
        x = self.conv2(x)
        # print(f"conv2 : {x.shape}")
        x = self.conv3(x)
        # print(f"conv3 shape : {x.shape}")
        x = self.conv4(x)
        # print(f"conv4 shape : {x.shape}")
        x = self.conv5(x)
        # print(f"conv5 shape : {x.shape}")
        x = self.conv6(x)
        # print(f"conv6 shape : {x.shape}")
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        # print(f"fc1 shape : {x.shape}")
        x = self.fc2(self.lrelu(x))
        # print(x.shape)
        return x
    
    
    
# if __name__ == "__main__":
#     image = torch.randn(1,3,448,448).to(device)
#     model = Yolov1(num_class=2).to(device)
#     out = model(image)
#     print(out.view(out.shape[0],7,7,-1).shape)