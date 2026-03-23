import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from PIL import Image

gold_data = pd.read_csv('gld_price_data.csv')
X = gold_data.drop(['Date', 'GLD'], axis=1)
y = gold_data['GLD']

print(X.shape," \n", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
print(X_train.shape, X_test.shape)

reg = RandomForestRegressor(n_estimators=200, random_state=2)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
score = r2_score(y_pred, y_test)


# web app
st.title("Gold price prediction website!")
img = Image.open('img.jpg')
st.image(img, width=200, use_column_width=True)

st.subheader("Using Random Forest Regressor")
st.write(gold_data)
st.subheader("Model Performence")
st.write(score)