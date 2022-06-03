from matplotlib.pyplot import imread
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn.model_selection
import torch.optim as optim
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification

text_file = open("processed_data/detection_camera1/lane_changes.txt")
lines = text_file.read().split('\n')

newlines = []
for line in lines:
  row = []
  for word in line.split():
    row.append(word)
  newlines.append(row)

newlines = [nl for nl in newlines if nl]

final = []
count = 0
for row in newlines:
  #print(row)
  if count < int(row[3]):
    while count < int(row[3]):
      final.append([count, 0])
      count += 1
  if count >= int(row[3]) and count <= int(row[5]):
    while count >= int(row[3]) and count <= int(row[5]):
      final_add = int(row[2])
      if final_add == 3:
        final_add = 1
      else:
        final_add = 2
      final.append([count, final_add])
      count += 1
  else:
    final.append([count, 0])
    count += 1
while count <= 23844:
  final.append([count, 0])
  count += 1

print(len(final))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224', num_labels = 3)

class PreventionImageDataset(Dataset):
    """The training table dataset.
    """
    def __init__(self, x_path):
        x_filenames = glob(x_path + '*.jpg') # Get the filenames of all training images
        #self.x_data = x_filenames
        #self.x_data = [torch.from_numpy(imread(filename)) for filename in x_filenames] # Load the images into torch tensors
        self.x_data = []
        self.y_data = []
        for filename in x_filenames:
          cur_ind = int(filename.split(".")[0].split("s")[1])
          if cur_ind <= 23834 and cur_ind >= 9:
            self.x_data.append(filename)
            self.y_data.append(final[cur_ind + 10][1])

        #self.y_data = [final[int(filename.split(".")[0].split("s")[1])][1] for filename in x_filenames]
        #self.y_data = target_label_list # Class labels
        self.len = len(self.x_data) # Size of data
        
    def __getitem__(self, index):
        image_tensors = []
        for i in range(index-9, index+1):   
          img = imread(self.x_data[i])
          scale_percent = 50
          #scale_percent = 25 # percent of original size
          width = int(img.shape[1] * scale_percent / 100)
          height = int(img.shape[0] * scale_percent / 100)
          dim = (width, height)
  
          # resize image
          resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
          #encoding = feature_extractor(images=resized, return_tensors="pt")
          image_tensors.append(torch.from_numpy(resized))
        image_tensors = torch.cat(image_tensors, dim=1)
        image_tensors = image_tensors
        encodings = feature_extractor(images=image_tensors, return_tensors="pt")
        return encodings['pixel_values'].to(device), torch.tensor(self.y_data[index]).to(device)
        
    def __len__(self):
        return self.len

dataset = PreventionImageDataset("VideoData/VideoData/")

net = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
net.cuda()
net.train()

def reset_weights(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    m.reset_parameters()

kfold = sklearn.model_selection.KFold(n_splits=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs = 11

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc

def train_test_dataset(dataset, test_split=0.167, random_state=42):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=42)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, val_idx)
    return datasets

#for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
#  print('------------fold no---------{}----------------------'.format(fold))
  #train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
  #test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
datasets = train_test_dataset(dataset)
trainloader = torch.utils.data.DataLoader(
                      datasets['train'], 
                      batch_size=32)
testloader = torch.utils.data.DataLoader(
                      datasets['test'],
                      batch_size=32)

net.apply(reset_weights)
#fcount = 0
 
for epoch in range(epochs):
  print("Epoch Number: " + str(epoch))
  #running_loss = 0.0
  train_acc = 0.0
  itercount = 0
  for i, data in enumerate(trainloader, 0):
    if i == 313:
      break
    print("Iteration Number: " + str(i))
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
        #with torch.no_grad():
    #inputs = torch.swapaxes(inputs, 1, 3)
    #inputs = torch.swapaxes(inputs, 2, 3)
    #print(inputs.size())
        #labels = torch.tensor(labels)
    inputs.cuda()
    inputs = torch.reshape(inputs, [32, 3, 224, 224])
    labels.cuda()
    outputs = net(inputs)
    outputs = outputs.logits.cuda()

    torch.save(outputs, 'vit_tensors_tenv1/' + str(epoch) + '_' + str(i) + '_outputs.pt')
    torch.save(labels, 'vit_tensors_tenv1/' + str(epoch) + '_' + str(i) + '_labels.pt')
   
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

        # print statistics
    running_loss = loss.item()
    train_acc += multi_acc(outputs, labels)
    itercount += 1
        #if i % 2000 == 1999:    # print every 2000 mini-batches
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1:.3f}')
    #running_loss = 0.0
    torch.cuda.empty_cache()
    print(f'[{epoch + 1}, {i + 1:5d}] Train Acc: {train_acc/itercount:.3f}')
  #running_loss = 0.0
  torch.save(net.state_dict(), "ckpts_vit_tenv1/" + str(epoch))
  #fcount += 1

print('Finished Training')
