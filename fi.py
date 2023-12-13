import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

# Streamlitアプリの設定
st.title("手書き数字画像認識アプリ")

# 画像アップロード
# 学習済みモデルのロード


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
