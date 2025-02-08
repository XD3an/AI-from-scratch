import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


# define image transformer (image to tensor)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# load datasets (MNIST)
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# define CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)  # image size
        self.fc2 = nn.Linear(64, 10)  # 10 neuron for 0-9
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)  # flattern
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# initialize the model
model = CNN_Model()

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ttrain the model
for epoch in range(5):  # 5 epoch
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data in trainloader:
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Accuracy: {100 * correct / total}%")
    
# evaluation the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test accuracy: {100 * correct / total}%")

# prediction
model.eval()
with torch.no_grad():
    for i in range(5):
        img, label = testset[i]
        img = img.unsqueeze(0)
        output = model(img)
        _, predicted = torch.max(output, 1)
        print(f"Prediction: {predicted.item()}, True Label: {label}")

plt.imshow(testset[0][0].squeeze(), cmap=plt.cm.binary)
plt.title(f"Prediction: {predicted.item()}, True Label: {label}")
plt.show()

