import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv("house_data.csv")

X=data[["Rooms","Size"]]
y=data["Price"]

model=LinearRegression()

model.fit(X,y)

rooms=int(input("Enter number of rooms: "))
size=int(input("Enter house size: "))

prediction=model.predict([[rooms,size]])

print("Predicted House Price:")

print(prediction[0])