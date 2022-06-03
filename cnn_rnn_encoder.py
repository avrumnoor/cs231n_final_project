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
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

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
        return image_tensors, torch.tensor(self.y_data[index]).to(device), 1
        
    def __len__(self):
        return self.len

dataset = PreventionImageDataset("VideoData/")

print(len(dataset))

train_loader = DataLoader(dataset=dataset,
                         batch_size=32,
                         shuffle=True,
                         num_workers=2
)

#model from https://github.com/Lin-Zhipeng/CNN-RNN-A-Unified-Framework-for-Multi-label-Image-Classification

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = torch.reshape(self.embed(captions), (32,1,3))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        #print(packed.size())
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image featuresi using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


encodernet = EncoderCNN(3)
decodernet = DecoderRNN(3, 1024, 3, 2)
encodernet.cuda()
decodernet.cuda()

epochs = 5

#kfold = sklearn.model_selection.KFold(n_splits=6)
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(decodernet.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
params = list(decodernet.parameters())# + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.01)

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

 
for epoch in range(epochs):
  print("Epoch Number: " + str(epoch))
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    print("Iteration Number: " + str(i))
    if i == 313:
        break
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels, lengths = data
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
        #with torch.no_grad():
    inputs = torch.swapaxes(inputs, 1, 3)
    inputs = torch.swapaxes(inputs, 2, 3)
        #labels = torch.tensor(labels)
    inputs.cuda()
    labels.cuda()
    encoderOutputs = encodernet.forward(inputs.float())
    outputs = decodernet.forward(encoderOutputs, labels, lengths)  
    #print(outputs)
    outputs.cuda()
        #outputs = torch.reshape(outputs, (32,))
        #labels = torch.tensor(labels, dtype=torch.float32)
        #print(outputs.size())
        #print(labels.size())
    torch.save(outputs, "tensors_outputs/" + str(epoch) + "_" + str(i) + "_outputs.pt")
    torch.save(labels, "tensors_labels/" + str(epoch) + "_" + str(i) + "_labels.pt")
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

        # print statistics
    running_loss += loss.item()
    train_acc = multi_acc(outputs, labels)
    #if i % 2000 == 1999:    # print every 2000 mini-batches
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1:.3f} | Train Acc: {train_acc:.3f}')
    running_loss = 0.0
    torch.cuda.empty_cache()
  torch.save(decodernet.state_dict(), "ckpts_dec/" + str(epochs))
  torch.save(encodernet.state_dict(), "ckpts_enc/" + str(epochs))
  #fcount += 1

print('Finished Training')
