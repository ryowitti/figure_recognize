import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

figure = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 学習済みモデルの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.bn2(self.conv5(x)))
        x = self.pool(x)
        x = x.view(x.size(0), 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# モデルのロード
model = Net()
model.load_state_dict(torch.load("C:/model_f_cnn.pth", map_location=torch.device('cpu')))

img_size = 28

def predict(img):
    # モデルへの入力
    img = img.convert("L")  # モノクロに変換
    img = img.resize((img_size, img_size))  # サイズを変換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))  # チャンネル数が1なのでタプル内に1つの要素を持つ
    ])
    img = transform(img)
    x = img.unsqueeze(0)  # バッチ次元を追加

    # 予測
    model.eval()
    y = model(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)

    # 正しいクラスラベルの順序に修正
    sorted_figure = [figure[idx] for idx in sorted_indices]

    return list(zip(sorted_figure, sorted_prob.tolist()))

# Streamlitアプリの設定
st.title("手書き数字画像認識アプリ")

# 画像アップロード
img_file = st.file_uploader("手書き数字の画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# 予測ボタンの追加
if img_file is not None:
    with st.spinner("認識中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        results = predict(img)

        # 結果の表示
        st.subheader("判定結果")
        n_top = 3  # 確率が高い順に3位まで返す
        for result in results[:n_top]:
            st.write(f"{result[0]}: {round(result[1] * 100, 2)}%")
