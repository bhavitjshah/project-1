import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.subheader('''Diabetes Detection:''')
image = Image.open(r'D:\project\Image_1.png')
if image.mode != 'RGB':
    image = image.convert('RGB')
st.image(image, use_column_width=True)

# dataset#
df = pd.read_csv(r'D:\project\diabetes dataset.csv')
st.subheader('Data Information:')
st.dataframe(df)

# To show statistics on data
st.write(df.describe())
chart = st.bar_chart(df)

# split data in X independent variable and Y dependent variable
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values


# get features input from user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 115)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 20)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 20, 85, 25)
    # store dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }
    # Transform the data into a data frame
    feature = pd.DataFrame(user_data, index=[0])
    return feature


# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the user input
st.subheader('This displays the User Input:')
st.write(user_input)

# Create and train model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the model matrices
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store the models prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
if prediction == 0:
    st.subheader('Diabetes is not detected')
else:
    st.subheader('Diabetes is detected')
