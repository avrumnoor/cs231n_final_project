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
        for i in range(index-3, index+1):
          img = imread(self.x_data[i])
          scale_percent = 25 # percent of original size
          width = int(img.shape[1] * scale_percent / 100)
          height = int(img.shape[0] * scale_percent / 100)
          dim = (width, height)
  
          # resize image
          resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
          image_tensors.append(torch.from_numpy(resized).to(device))
        image_tensors = torch.cat(image_tensors, dim=1)
        image_tensors = image_tensors.to(device)
        return image_tensors, torch.tensor(self.y_data[index]).to(device)
        
    def __len__(self):
        return self.len

dataset = PreventionImageDataset("VideoData/VideoData/")

#print(dataset.size())

train_loader = DataLoader(dataset=dataset,
                         batch_size=32,
                         shuffle=True,
                         num_workers=2
)


class PrevNet(nn.Module):
    def __init__(self):
        super(PrevNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9024, 128)
        self.fc2 = nn.Linear(128, 3)
 
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 40)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

net = PrevNet()
net.cuda()

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

def train_test_dataset(dataset, test_split=0.167):
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
    print("Iteration Number: " + str(i))
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
        #with torch.no_grad():
    inputs = torch.swapaxes(inputs, 1, 3)
    inputs = torch.swapaxes(inputs, 2, 3)
    #print(inputs.size())
        #labels = torch.tensor(labels)
    inputs.cuda()
    labels.cuda()
    outputs = net(inputs.float())
    outputs.cuda()
        #outputs = torch.reshape(outputs, (32,))
        #labels = torch.tensor(labels, dtype=torch.float32)
        #print(outputs.size())
        #print(labels.size())
    torch.save(outputs, 'cnn_tensors/' + str(epoch) + '_' + str(i) + '_outputs.pt')
    torch.save(labels, 'cnn_tensors/' + str(epoch) + '_' + str(i) + '_labels.pt')
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
  torch.save(net.state_dict(), "ckpts/" + str(epoch))
  #fcount += 1

print('Finished Training')
