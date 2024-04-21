import os
import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
from tempfile import TemporaryDirectory



def set_variables(a, b):
    global path, user_name
    path, user_name = a, b

def see_variable():
    global path, user_name
    return path, user_name

def train_face_model():
    transformer = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformer_classify = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dat = ImageFolder(os.path.join(path+'\\data',user_name + '\\train'), transform=transformer)
    test_dat = ImageFolder(os.path.join(path+'\\data', user_name + '\\test'), transform=transformer_classify)
    data_len = [len(train_dat), len(test_dat)]
    train_dat = DataLoader(train_dat, batch_size=4, shuffle=True)
    test_dat = DataLoader(test_dat, batch_size=4, shuffle=True)
    data_load = [train_dat, test_dat]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    num_ftrs = model2.fc.in_features
    model2.fc = torch.nn.Linear(num_ftrs, 1)
    model2 = model2.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    train_model(path,user_name,device,model2,criterion,
                           optimizer_ft,exp_lr_scheduler,
                           data_len,data_load,num_epochs=25)

    return 'success'

def train_model(path,user,device,model,criterion,optimizer,scheduler,data_len,data_load,num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path =os.path.join(path+'\\data\\'+user,user+"_best_model_params.pt")
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in range(2):
                if phase ==0:
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in data_load[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase==0):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase ==0:
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase ==0:
                    scheduler.step()

                epoch_loss = running_loss / data_len[phase]
                epoch_acc = running_corrects.double() / data_len[phase]
                kind='Train' if phase==0 else 'Test'
                print(f'{kind} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase ==1 and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
            #print('end of 2nd loop')

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path))
        torch.save(model.state_dict(),best_model_params_path)

    return 'success'
