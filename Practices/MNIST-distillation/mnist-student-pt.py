import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary

# define Teacher Model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784,  1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200,   10)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x

# define Student Model (smaller model)
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20,  20)
        self.fc3 = nn.Linear(20,  10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    '''
    Soft target loss (KLDivLoss):

        q_i = \frac{exp(z_i / T)}{\sum_j exp(z_j / T)}

    Hard target loss (CrossEntropyLoss):

        cross_entropy(student_logits, labels)

    '''
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1)
    )
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

def test_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def main():
    # define image transformer (image to tensor)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # load datasets (MNIST)
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # initialize the models
    student_model_no_distillation = StudentModel()
    student_model_with_distillation = StudentModel()
    teacher_model = TeacherModel()
    teacher_model.load_state_dict(torch.load('mnist-teacher-model.pth'))
    summary(teacher_model)
    summary(student_model_with_distillation)

    # define optimizer for both student models
    optimizer_no_distillation = optim.Adam(student_model_no_distillation.parameters(), lr=1e-4)
    optimizer_with_distillation = optim.Adam(student_model_with_distillation.parameters(), lr=1e-4)

    # track the loss and accuracy for plotting
    loss_no_distillation = []
    accuracy_no_distillation = []
    loss_with_distillation = []
    accuracy_with_distillation = []

    # train both models
    for epoch in range(5):  # 5 epochs
        # Train model without distillation
        student_model_no_distillation.train()
        running_loss_no_distillation = 0.0
        correct_no_distillation = 0
        total_no_distillation = 0
        for images, labels in trainloader:
            optimizer_no_distillation.zero_grad()
            logits_no_distillation = student_model_no_distillation(images)
            loss_no_distillation_epoch = F.cross_entropy(logits_no_distillation, labels)
            loss_no_distillation_epoch.backward()
            optimizer_no_distillation.step()

            running_loss_no_distillation += loss_no_distillation_epoch.item()
            _, predicted = torch.max(logits_no_distillation, 1)
            total_no_distillation += labels.size(0)
            correct_no_distillation += (predicted == labels).sum().item()

        loss_no_distillation.append(running_loss_no_distillation / len(trainloader))
        accuracy_no_distillation.append(100 * correct_no_distillation / total_no_distillation)

        # Train model with distillation
        student_model_with_distillation.train()
        running_loss_with_distillation = 0.0
        correct_with_distillation = 0
        total_with_distillation = 0
        for images, labels in trainloader:
            optimizer_with_distillation.zero_grad()
            student_logits = student_model_with_distillation(images)
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            # calculate distillation loss (soft, hard)
            loss_with_distillation_epoch = distillation_loss(
                student_logits, 
                teacher_logits, 
                labels, 
                temperature=7.0, 
                alpha=0.3)
            loss_with_distillation_epoch.backward()
            optimizer_with_distillation.step()

            running_loss_with_distillation += loss_with_distillation_epoch.item()
            _, predicted = torch.max(student_logits, 1)
            total_with_distillation += labels.size(0)
            correct_with_distillation += (predicted == labels).sum().item()

        loss_with_distillation.append(running_loss_with_distillation / len(trainloader))
        accuracy_with_distillation.append(100 * correct_with_distillation / total_with_distillation)

        print(f"Epoch {epoch+1} (without distillation) Loss: {loss_no_distillation[-1]:.4f}, Accuracy: {accuracy_no_distillation[-1]:.2f}%")
        print(f"Epoch {epoch+1} (with    distillation) Loss: {loss_with_distillation[-1]:.4f}, Accuracy: {accuracy_with_distillation[-1]:.2f}%")

    # Test both models on test data
    accuracy_no_distillation_test = test_model(student_model_no_distillation, testloader)
    accuracy_with_distillation_test = test_model(student_model_with_distillation, testloader)

    print(f"Test Accuracy (without distillation): {accuracy_no_distillation_test:.2f}%")
    print(f"Test Accuracy (with distillation): {accuracy_with_distillation_test:.2f}%")

    # Plot training loss and accuracy
    plt.figure(figsize=(12, 6))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(loss_no_distillation, label='without distillation', color='blue')
    plt.plot(loss_with_distillation, label='with distillation', color='red')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_no_distillation, label='without distillation', color='blue')
    plt.plot(accuracy_with_distillation, label='with distillation', color='red')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()

