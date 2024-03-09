# vision_maze
openCV 를 활용해 손을 움직여 미로를 푸는 ‘시니어를 위한 치매예방게임: 비전 미로

## 개발 배경
대한노인정신의학회에 따르면 전 세계적으로 치매는 65세 이상 노인에서 약 5-10%의 유병률을 보이고 있고, 새로운 치매 환자가 연간 460만 명, 7초당 한 명씩 발생합니다. 치매의 유병률은 65세를 기준으로 5세가 증가할 때 마다 거의 2배씩 증가하였습니다. 급속한 고령화로 우리나라 65세 이상 노인 인구의 치매 유병률은 계속 상승할 것으로 전망되었으며, 환자수도 2050년까지 20년 마다 2배씩 증가하여 2010년 약 47만 명, 2030년 약 114만 명, 2050년에는 213만 명으로 추정됩니다. 치매 예방 프로그램에 대한 관심이 높아지고 있는 상황 속에서, 저희 팀은 게임이 치매 예방에 도움이 될 수 있다는 사실에 주목하였습니다. 여러 논문에 따르면, 게임은 시각, 청각 등을 자극해 뇌 기능이 유지되도록 돕고 오감을 자극하고 손을 사용하게 만들어 적절한 자극을 줄 수 있어 망가진 뇌 기능을 회복하는데 도움이 됩니다. 게임은 운동보다 부상 위험이 적고, 몸이 불편하더라도 새로운 경험을 계속 할 수 있어 유럽 같은 선진국에서는 노년층 치매치료에 쓰이고 있습니다. 우리 나라 역시 문체부와 게임문화재단에서 조손가정 보호자의 게임인식 개선과 올바른 게임 이용 지도를 돕는 고령층 대상 교육을 전국에서 실시한 바 있습니다.
따라서 저희 팀은 openCV를 활용해 손을 움직여 미로를 푸는 ‘시니어를 위한 치매예방게임: 비전 미로’를 개발하게 되었습니다.


## 개발 환경
- window10
- intel core i7
- Nvidia GeForce RTX 4070
- openCV 4.8
- python 3.8
- Yolo v7
  
## 학습 과정
Yolo v7 은 Yolo v5 보다 최신 모델로, 높은 정확도와 좋은 추론 속도를 제공합니다. 따라서 Yolo v7 으로 학습을 진행하였습니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/d891f374-8204-42bf-9c3e-da9d83294eef)

class는 총 5가지로, 👈(left), 👉(right), 👆(up), 👇(down), 👊(fist)로 이루어져 있습니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/09269aa0-d0b6-446e-80c9-b7d19a1ccb78)
![image](https://github.com/MIN60/vision_maze/assets/49427080/e780ea08-fe70-45c0-9218-7894a8c9c901)

roboflow에서 라벨링을 통해 데이터셋 총 16,642장으로 학습을 진행하였습니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/970815df-c060-4fa5-b601-38f5c86485b7)

데이터셋을 만들 때 data Augmentation을 이용하였고 rotation, Shear, Brightness조정을 통해 더 다양한 데이터셋을 만들고자 하였습니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/2d7689f7-b4a1-496a-85a9-3069f7b0ca89)

nvidia 4070 gpu를 이용하여 학습을 진행하였습니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/fa82da45-73f1-4b52-a596-9ab70e5b682b)

anaconda로 가상환경을 만들어 python 3.8환경을 만들었습니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/bd8cfece-94ae-4abc-835e-666af27c18fa)

배치사이즈 16에 epoch 50으로 학습을 진행하였습니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/474b882a-eb21-4bc4-acd0-82d0c531e5b5)

data.yaml 파일입니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/7d4589de-4239-4d0d-a349-a53ce4a3c5e6)

학습 완료 후 결과로 pt파일이 만들어졌습니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/5cd4484b-ad68-4df6-9308-e59961e0338d)

## 학습 결과

![image](https://github.com/MIN60/vision_maze/assets/49427080/06bd65c7-4472-4c6d-9f6f-575e06f72951)
![image](https://github.com/MIN60/vision_maze/assets/49427080/9b44be75-3aa7-49f2-b036-3ac161540860)


## 미로 게임 개발

### 메인 화면

![image](https://github.com/MIN60/vision_maze/assets/49427080/052f9ecd-6dcd-4384-b793-6b676c693822)

메인 화면입니다. 클릭해서 시작할 수 있으며, 0번을 누르면 도움말이 뜹니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/32c96ae7-fbf9-452e-82f5-c095ed4206b7)

게임 도움말 입니다.

### 스테이지 선택

![image](https://github.com/MIN60/vision_maze/assets/49427080/50467fb5-5a29-4c6e-9953-d24209e5c87b)

1~3번을 입력하여 스테이지를 선택할 수 있습니다.

### 게임 화면

![image](https://github.com/MIN60/vision_maze/assets/49427080/da11d768-1c5f-46a8-8244-ff236ed0cf40)

게임 화면입니다.

![image](https://github.com/MIN60/vision_maze/assets/49427080/f8a3a2b1-f706-47e8-9733-a2d3be6c34eb)

출구에 도달하면 축하 이미지가 뜹니다.







