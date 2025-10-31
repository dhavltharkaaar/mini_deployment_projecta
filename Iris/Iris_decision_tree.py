from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import streamlit as st


# Bold headline
st.markdown("**Iris Prediction Using Decision Tree**")

iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Streamlit inputs
sepal_length = st.number_input("Sepal length (cm)")
sepal_width  = st.number_input("Sepal width (cm)")
petal_length = st.number_input("Petal length (cm)")
petal_width  = st.number_input("Petal width (cm)")

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = clf.predict(features)
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    st.success(f"Predicted species: {species_map[prediction[0]]}")
