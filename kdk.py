import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import models
import torch.nn.functional as F
from torchvision.models import resnet
from models import model_sets

class Kdk():

    def __init__(self, train_loader, val_loader, dataset, lr):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset = dataset
        self.lr = lr
    def train_teacher(self, teacher, train_loader, val_loader, device, dataset):
        if self.dataset == 'Yahoo':
            self.teacher_epoch = 10
        else:
            self.teacher_epoch = 20
        self.teacher.train()
        criterion = nn.CrossEntropyLoss()
        if not self.dataset == 'Yahoo':
            optimizer = optim.Adam(self.teacher.parameters(), lr=0.0001)
        else:
            optimizer = optim.SGD(
                    [
                        {"params": self.teacher.mixtext_model.bert.parameters(), "lr": 5e-6},
                        {"params": self.teacher.mixtext_model.linear.parameters(), "lr": 5e-4},
                    ],
                    lr=self.lr)
        stone1 = int(self.teacher_epoch * 0.5)
        stone2 = int(self.teacher_epoch * 0.8)
        lr_scheduler_teacher = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[stone1, stone2], gamma=args.step_gamma)
        for epoch in range(self.teacher_epoch):
            print('Epoch {}/{}'.format(epoch+1, self.teacher_epoch))
            print('-' * 10)
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in self.train_loader:
                if self.dataset == 'BCW' or self.dataset == 'Criteo':
                    inputs = inputs.float()
                elif self.dataset == 'Yahoo':
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].long().cuda()
                    labels = labels[0].long().cuda()
                _, inputs = split_data(inputs)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.teacher(inputs)
                outputs = F.log_softmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            lr_scheduler_teacher.step()
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = running_corrects.double() / len(self.train_loader)

            print('{} loss: {:.4f}, acc: {:.4f}'.format('Train accuracy',
                                                            epoch_loss,
                                                            epoch_acc))
        
        self.teacher.eval()

        correct = 0
        total = 0
        for data, target in self.val_loader:
            _, data = split_data(data)
            if self.dataset == 'BCW' or self.dataset == 'Criteo':
                data = data.float()
            elif self.dataset == 'Yahoo':
                for i in range(len(target)):
                    data[i] = data[i].long().cuda()
                target = target[0].long().cuda()
            data, target = data.to(self.device), target.to(self.device)
            output = self.teacher(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Calculate validation accuracy
        val_accuracy = 100 * correct / total
        print('Validation Accuracy: {:.2f}%'.format(val_accuracy))
        torch.save(self.teacher.state_dict(), f'saved_experiment_results/teacher_models/{self.dataset}')


    def train_teacher(self, train_loader, val_loader, dataset):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if dataset == 'CIFAR10' or dataset == 'CINIC10L':
            self.teacher = resnet.resnet50(pretrained=True)
            self.teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
            self.teacher.fc = nn.Linear(2048,10)
        elif dataset == 'BCW':
            self.teacher = model_sets.BottomModelForBcw()
            #self.teacher.fc3 = nn.Linear(20,10)
        elif dataset == 'Criteo':
            self.teacher = model_sets.BottomModelForCriteo(args.half, is_adversary=False)
            self.teacher.fc3 = nn.Linear(16, 2)
        elif dataset=='CIFAR100':
            self.teacher = resnet.resnet50(pretrained=True)
            self.teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
            self.teacher.fc = nn.Linear(2048,100)
        elif dataset == 'TinyImageNet':
            self.teacher = resnet.resnet50(pretrained=True)
            self.teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
            self.teacher.fc = nn.Linear(2048,200)
        elif dataset == 'Yahoo':
            self.teacher = model_sets.BottomModelForYahoo(10)
        self.teacher.to(self.device)
        
        if os.path.exists(f'saved_experiment_results/teacher_models/{dataset}'):
            self.teacher.load_state_dict(torch.load(f'saved_experiment_results/teacher_models/{dataset}'))
        else:
            self.teacher = self.train_teacher()
        
        self.teacher.eval()
    
    def get_topk(self, targets, k, h_tragets, epsilon=0.65):
        #print(targets.shape)
        _, indices = torch.topk(targets, k, dim=1)
        top_targets = torch.zeros(targets.shape)
        for i, h in zip(range(targets.shape[0]), h_tragets):
            prob_sum = sum([targets[i, ind] for ind in indices[i] if not ind == h])
            for j in indices[i]:
                top_targets[i, j] = epsilon /(k-1)
        
        for i,j in zip(range(targets.shape[0]), h_tragets):
            top_targets[i, j] = 1 - epsilon
        
        #print(top_targets)
        return top_targets
    
    def soft_labels(self, data, target):
        target_soft = self.teacher(data).detach()
        target_soft = F.softmax(target_soft, dim=1)
        target_soft = self.get_topk(target_soft, self.top_k, target, args.soft_epsilon).to(self.device)
        return target_soft