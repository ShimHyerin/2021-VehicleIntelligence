# Pytorch Pre-trained Model 성능 비교
pytorch - torchvision 에서 제공하는 4가지 모델의 top-1 accuracy와 top-5 accuracy를 구해 성능을 비교하였다.

---
- DataSet: ImageNet Validation set
- models
```
  1. AlexNet
  2. VGG16
  3. ResNet18
  4. GoogLeNet
 ```
 
 ## Code >> [Go](https://github.com/ShimHyerin/2021-VehicleIntelligence/blob/main/HW2/modelCompare.py)
 모델 성능 비교 코드입니다. 코드 실행시 실행 결과를 저장한 txt 파일 및 그래프 사진이 생성됩니다.
 ``` python
 # rank 1
  _, pred = torch.max(outputs, 1)
  total += labels.size(0)
  top1 += (pred == labels).sum().item()
 ```
 (code example)
 ## Report >> [Go](https://github.com/ShimHyerin/2021-VehicleIntelligence/blob/main/HW2/modelCompareReport.pdf)
 위 4가지 모델에 대한 조사 및 전체 코드와 실행 결과 내용이 담겨져 있습니다.
 
 ---
 ### Result
top-1 accuracy 성능 비교
> VGG16 (71.59%) > googleNet (69.78%) > ResNet(69.76%) > AlexNet (56.52%)

Result Graph :: > [Code](https://github.com/ShimHyerin/2021-VehicleIntelligence/blob/main/HW2/resGraph.py)
<img src="https://github.com/ShimHyerin/2021-VehicleIntelligence/blob/main/HW2/modelCompareGraph.png" width="60%">


모델 성능 테스트 과정 및 결과 :: > [과정기록txt](https://github.com/ShimHyerin/2021-VehicleIntelligence/blob/main/HW2/modelCompareResFin.txt) // [최종결과txt](https://github.com/ShimHyerin/2021-VehicleIntelligence/blob/main/HW2/resFinal.txt)
![image](https://user-images.githubusercontent.com/54926467/115361128-3d2aa400-a1fb-11eb-9ad8-e0458b7600bc.png)


>model :: AlexNet
>-----------------------------
>top-1 accuracy :: 56.52%
>top-5 accuracy :: 79.07%


>model :: VGG16
>-----------------------------
>top-1 accuracy :: 71.59%
>top-5 accuracy :: 90.38%


>model :: ResNet
>-----------------------------
>top-1 accuracy :: 69.76%
>top-5 accuracy :: 89.08%


>model :: googLeNet
>-----------------------------
>top-1 accuracy :: 69.78%
>top-5 accuracy :: 89.53%
