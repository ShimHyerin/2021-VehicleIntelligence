import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import sys
from matplotlib import pyplot as plt

sys.stdout = open('modelCompareResFin.txt', 'w')

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # dataset path
    valdir = '/home/hyerin/modelCompare/ImageNet/val'
    # dataset load
    val_set = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ]))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True, num_workers=4)


    resTop1 = []
    resTop5 = []

    # models
    alexnet = torchvision.models.alexnet(pretrained=True).to(device)
    vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
    resnet18 = torchvision.models.resnet18(pretrained=True).to(device)
    googlenet = torchvision.models.googlenet(pretrained=True).to(device)
    models_name = ['AlexNet', 'VGG16', 'ResNet', 'googleNet']
    models = [alexnet, vgg16, resnet18, googlenet]


    for i in range(4):
        print("\n---------Model :: {} ---------\n".format(models_name[i]))
        model = models[i]
        model.eval()

        top1 = 0
        top5 = 0
        total = 0

        with torch.no_grad():
            for idx, (images, labels) in enumerate(val_loader):

                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images) # batch_size eval

                # rank 1
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                top1 += (pred == labels).sum().item()


                # rank 5
                _, rank5 = outputs.topk(5, 1, True, True)
                rank5 = rank5.t()
                correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))
                correct5 = correct5.contiguous()

                for k in range(6):
                    correct_k = correct5[:k].view(-1).float().sum(0, keepdim=True)
                top5 += correct_k.item()

                print("step : {} / {}".format(idx + 1, len(val_set)/int(labels.size(0))))
                print("top-1 percentage :  {0:0.2f}%".format(top1 / total * 100))
                print("top-5 percentage :  {0:0.2f}%".format(top5 / total * 100))
        
        print("\n---------Result :: {} ---------\n".format(models_name[i]))
        print("top-1 percentage :  {0:0.2f}%".format(top1 / total * 100))
        print("top-5 percentage :  {0:0.2f}%".format(top5 / total * 100))
        print("---------------------------------------\n\n")

        # res store
        resTop1.append(top1/total*100)
        resTop5.append(top5/total*100)
    
    sys.stdout.close()

    # output store
    f = open('allResFin.txt', 'w')

    for i in range(4):
        print('model :: {}'.format(models_name[i]), file=f)
        print('-----------------------------\n',file=f)
        print('top-1 accuracy :: {0:0.2f}%'.format(resTop1[i]), file=f)
        print('top-5 accuracy :: {0:0.2f}%'.format(resTop5[i]), file=f)
        print('\n-----------------------------\n\n\n',file=f)

    f.close()

# draw Graph
def create_x(t, w, n, d): # numberOfData, BarWidth, numOfCurrentData, numOfDataLen
    return [t*x + w*n for x in range(d)]
top1_x = create_x(2, 0.8, 1, 4)
top5_x = create_x(2, 0.8, 2, 4)
ax = plt.subplot()
ax.bar(top1_x, resTop1, color='salmon', label='top-1 accuracy')
ax.bar(top5_x, resTop5, color='silver', label='top-5 accuracy')
x = [(a+b)/2 for (a,b) in zip(top1_x, top5_x)]
ax.set_xticks(x)
ax.set_xticklabels(models_name)
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy percentage')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
plt.savefig('modelCompareGraph.png', format='png', dpi=300)
plt.show()