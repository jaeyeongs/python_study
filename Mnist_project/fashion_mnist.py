# 필요한 패키지 불러오기
from torchvision import datasets, transforms, utils
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np

# transform 도구 사용 - ToTensor() : 텐서로 바꿔주는 함수
transform = transforms.Compose([
    transforms.ToTensor()
])

# 학습 & 테스트 데이터 다운로드 및 불러오기
trainset = datasets.FashionMNIST(
    root = './.data/',
    train = True,
    download = True,
    transform = transform
)

testset = datasets.FashionMNIST(
    root = './.data/',
    train = True,
    download = True,
    transform = transform
)

# batch : 한 번에 처리하는 데이터 개수
batch_size = 16   # 이미지를 반복마다 16개씩 읽어줌

train_loader = data.DataLoader(
    dataset = trainset,
    batch_size = batch_size
)

test_loader = data.DataLoader(
    dataset = testset,
    batch_size = batch_size
)

dataiter = iter(train_loader)    # iter() 함수를 이용하여 반복문 안에서 이용할 수 있게 함
images, labels = next(dataiter)  # next() 함수를 이용하여 배치 1개 가져옴

img = utils.make_grid(images, padding = 0)  # 여러 이미지를 한 번에 보여줌
npimg = img.numpy()                         # 이미지가 텐서이므로 matplotlib과 호환하기 위해 넘파이 행렬 변환
plt.figure(figsize=(10, 7))
plt.imshow(np.transpose(npimg, (1,2,0)))    # 0번째 차원을 맨 뒤로 보냄
plt.show()

# 가져온 16개 레이블 출력
print(labels)

# 클래스별로 숫자를 부여
CLASSES = {
    0 : 'T-shirt/top',
    1 : 'Trouser',
    2 : 'Pullover',
    3 : 'Dress',
    4 : 'Coat',
    5 : 'Sandal',
    6 : 'Shirt',
    7 : 'Sneaker',
    8 : 'Bag',
    9 : 'Ankle boot'
}

# 생성한 딕셔너리로 labels 출력
for label in labels:
    index = label.item()
    print(CLASSES[index])


idx = 1                                    # FashionMNIST 데이터셋의 첫 번째 이미지
item_img = images[idx]
item_npimg = item_img.squeeze().numpy()    # matplotlib 하기 위해 행렬 변환
plt.title(CLASSES[labels[idx].item()])
plt.imshow(item_npimg, cmap='gray')
plt.show()

# 심층 인공 신경망(DNN : Deep Neural Network)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

# CUDA 사용 가능 여부
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# 데이터가 많아서 여러개의 배치로 잘라 사용
EPOCHS = 30 # 학습 데이터를 30번 봄
BATCH_SIZE = 64

# 모델 설계
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1440000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# 모델 선언
model = Net().to(DEVICE)

# 최적화(SGD : 확률적 경사하강법)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)   # 학습 데이터를 DEVICE의 메모리에 보냄
        optimizer.zero_grad()                               # 반복할 때마다 기울기 계산
        output = model(data)
        loss = F.cross_entropy(output, target)              # 클래스가 10개 이므로 교차 엔트로피 사용
        loss.backward()                                     # 기울기 계산
        optimizer.step()                                    # 가중치 수정

# 모델 평가
def evaluate(model, test_loader):
    model.eval()    # 모델 평가 모드
    test_loss = 0   # 테스트 오차와 예측이 맞은 수를 담을 변수를 0으로 초기화
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data) # 모델의 예측 값

            # 모든 오차 더하기
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            # 가장 큰 값을 가진 클래스가 모델의 예측
            # 예측과 정답을 비교하여 일치할 경우 correct에 1을 더함
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 전체 데이터셋에 대한 오차와 맞힌 개수의 합을 데이터 수로 나누어 평균을 구함
    # 정답 평균에 100을 곱해서 정확도 측정
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# epoch 마다 학습과 테스트셋을 이용한 검증을 반복 후 결과 출력
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch, test_loss, test_accuracy))

torch.save(model.state_dict(), '/Users/sinjaeyeong/PycharmProjects/pythonProject1')

import cv2
data_inf = cv2.imread('/Users/sinjaeyeong/PycharmProjects/pythonProject1/test1.png').reshape(1,600,800,3)
print("orignal_image size {}:".format(data_inf.shape))
data_info1 = cv2.imread('/Users/sinjaeyeong/PycharmProjects/pythonProject1/test1.png', cv2.IMREAD_GRAYSCALE)
print("orignal_image size {}:".format(data_info1.shape))
data_info2 = torch.tensor(data_inf).float().to(DEVICE)
outputs = model(data_info2)
print(outputs)
print(torch.argmax(outputs))
# data_inf_resized_scaled = data_inf_resized/255.0
# print(np.argmax(model.predict(data_inf_resized_scaled)))



