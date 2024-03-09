✨ If you need a README in Korean, please check the 'korean' branch.

✨한국어로 된 readme가 필요하시다면 'korean' branch를 확인해주세요

# vision_maze
A Dementia Prevention Game for Seniors: Vision Maze, Utilizing OpenCV to Solve Mazes by Hand Movements

## Development Background
According to the Korean Neuropsychiatric Association, dementia has a global prevalence of approximately 5-10% among seniors over the age of 65, with 4.6 million new cases annually, equating to one new case every 7 seconds. The prevalence of dementia nearly doubles with every 5-year increase in age after 65. Due to rapid aging, the prevalence of dementia among the elderly population in Korea is expected to continue rising, with the number of patients projected to double every 20 years, reaching approximately 470,000 in 2010, 1.14 million in 2030, and 2.13 million by 2050. In light of the growing interest in dementia prevention programs, our team has focused on the potential of games to aid in dementia prevention. Research suggests that games can stimulate the brain by engaging senses such as sight and hearing, encourage the use of hands, and provide appropriate stimulation to help restore damaged brain functions. Games pose a lower risk of injury than physical exercise and allow individuals with physical limitations to continue experiencing new activities. Consequently, games are used in advanced countries for elderly dementia care. In Korea, the Ministry of Culture, Sports, and Tourism and the Game Culture Foundation have conducted nationwide education for the elderly, aimed at improving the perception of games among guardians of grandparent-grandchild families and promoting proper game usage.

Therefore, our team has developed "A Dementia Prevention Game for Seniors: Vision Maze," utilizing OpenCV to solve mazes by hand movements.


## Development Environment
- window10
- intel core i7
- Nvidia GeForce RTX 4070
- openCV 4.8
- python 3.8
- Yolo v7
  
## Learning Process
YOLO v7 is a newer model than YOLO v5, providing higher accuracy and better inference speed. Therefore, training was conducted with YOLO v7.

![image](https://github.com/MIN60/vision_maze/assets/49427080/d891f374-8204-42bf-9c3e-da9d83294eef)

Classes consist of a total of 5: 👈 (left), 👉 (right), 👆 (up), 👇 (down), and 👊 (fist).

![image](https://github.com/MIN60/vision_maze/assets/49427080/09269aa0-d0b6-446e-80c9-b7d19a1ccb78)
![image](https://github.com/MIN60/vision_maze/assets/49427080/e780ea08-fe70-45c0-9218-7894a8c9c901)

Training was conducted with a dataset of 16,642 images labeled through Roboflow.

![image](https://github.com/MIN60/vision_maze/assets/49427080/970815df-c060-4fa5-b601-38f5c86485b7)

Data augmentation was utilized to create a more diverse dataset, including rotation, shear, and brightness adjustments.

![image](https://github.com/MIN60/vision_maze/assets/49427080/2d7689f7-b4a1-496a-85a9-3069f7b0ca89)

Training was conducted using an Nvidia 4070 GPU.

![image](https://github.com/MIN60/vision_maze/assets/49427080/fa82da45-73f1-4b52-a596-9ab70e5b682b)

A virtual environment was created with Anaconda for a Python 3.8 environment.

![image](https://github.com/MIN60/vision_maze/assets/49427080/bd8cfece-94ae-4abc-835e-666af27c18fa)

Training was conducted with a batch size of 16 and for 50 epochs.

![image](https://github.com/MIN60/vision_maze/assets/49427080/474b882a-eb21-4bc4-acd0-82d0c531e5b5)

This is data.yaml file.

![image](https://github.com/MIN60/vision_maze/assets/49427080/7d4589de-4239-4d0d-a349-a53ce4a3c5e6)

After training, a .pt file was generated as the result.

![image](https://github.com/MIN60/vision_maze/assets/49427080/5cd4484b-ad68-4df6-9308-e59961e0338d)

## Training Results

![image](https://github.com/MIN60/vision_maze/assets/49427080/06bd65c7-4472-4c6d-9f6f-575e06f72951)
![image](https://github.com/MIN60/vision_maze/assets/49427080/9b44be75-3aa7-49f2-b036-3ac161540860)


## vision_maze

### Main Screen

![image](https://github.com/MIN60/vision_maze/assets/49427080/052f9ecd-6dcd-4384-b793-6b676c693822)

This is the main screen. You can start the game by clicking, and pressing 0 brings up the help information.

![image](https://github.com/MIN60/vision_maze/assets/49427080/32c96ae7-fbf9-452e-82f5-c095ed4206b7)

Game Help, here you can find information on how to play the game, controls, and other useful tips.

### Stage Selection

![image](https://github.com/MIN60/vision_maze/assets/49427080/50467fb5-5a29-4c6e-9953-d24209e5c87b)

You can select a stage by inputting numbers 1 to 3.

### Main Screen

![image](https://github.com/MIN60/vision_maze/assets/49427080/da11d768-1c5f-46a8-8244-ff236ed0cf40)

This is the game screen.

![image](https://github.com/MIN60/vision_maze/assets/49427080/f8a3a2b1-f706-47e8-9733-a2d3be6c34eb)

When you reach the exit, a congratulatory image appears.

## Video
[![Video Label](http://img.youtube.com/vi/bLYTToW55a4/0.jpg)](https://youtu.be/bLYTToW55a4)






