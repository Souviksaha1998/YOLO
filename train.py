from engine import Yolov1
from dataset_ import DataCreation
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

classList = ["car","fire"]
cell_Prediction  = (2 * 5) + len(classList)

train_transform_ = transforms.Compose([transforms.ToTensor()])
train_data = DataCreation("z","z1",train_transform_)


train_dataloader = DataLoader(train_data,batch_size=2,shuffle=False)

model = Yolov1(num_class=len(classList)).to(device)


loss_mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)

for im , bbox in (train_dataloader):
    
    pred = model(im.to(device))
    
    
    
    # print(pred.view(pred.shape[0],7,7,-1).shape)
    ground_truth = bbox.to(device).permute(0,2,3,1)
    bbox_ground_truth = ground_truth[...,0:4]
    objectness_truth  = ground_truth[...,4:5]
    class_truth = ground_truth[...,5:]
    
    # print(bbox_ground_truth.shape)
    
    
    # finding one bbox per grid cell.. and we need to make those prediction sum upto one (classes)
    pred = pred.view(pred.shape[0],7,7,-1)
    class_list = pred[...,10:]
    # print(class_list.shape)
    # softmax the class list
    pred_classlist = torch.softmax(class_list,dim=-1) # classes only
    
    bbox1_pred = pred[...,0:4]
    bbox1_objscore_pred = pred[...,4:5]
    
    
    
    bbox2_pred = pred[...,5:9]
    bbox2_objscore_pred = pred[...,9:10]
    
    # print(pred_classlist.shape)
    # print(bbox1_objscore_pred.shape)
    # print(bbox2_pred.shape)
    
    # we got max probality which object present for each grid cell (batch x 7 x 7 x 1)
    max_pred_class_grid , index = torch.max(pred_classlist,dim=-1,keepdim=True)
    
    # print(index)
    
    # bboxs_concat = torch.cat((bbox1_pred,bbox2_pred),dim=-1)
    
    # print(bboxs_concat.shape)
    
    # now need to select which bbox has highest conf with class
    best_bbox1 = max_pred_class_grid * bbox1_objscore_pred
    best_bbox2 = max_pred_class_grid * bbox2_objscore_pred
    
    concat = torch.cat((best_bbox1,best_bbox2),dim=-1)
    best_box_objectness_score  , index_ = torch.max(concat,dim=-1,keepdim=True)

    best_bbox_pred = index_ * bbox1_pred + (1 - index_) * bbox2_pred
    
    loss_ms = loss_mse(best_bbox_pred, bbox_ground_truth)
    print(loss_ms)
    
    
    
    # print(best_bbox_pred.shape)

    # # If you want to keep the shape of the selected bounding box tensor consistent with the input shape:
    # best_bbox_pred = best_bbox_pred.view(pred.shape[0], 7, 7, 4)
    
    
    # print(best_box_objectness_score , index_)
    
    # print(max_pred_class_grid.shape)
    # print(bbox2_objscore_pred.shape)
    
    # scores = torch.cat(
    #     (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim=0
    # )
    # best_box = scores.argmax(0).unsqueeze(-1)
    # best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    
    # filter_bbox = np.
    
    
    ## selecting the best box out of two box per grid cell
    
    # calculate objectness score based on confidence and class probablity
    # best on the objectness score we will select one bbox of that grid
    
    
    
    # print(pred)