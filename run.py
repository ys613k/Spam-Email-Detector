import streamlit as st
import pickle

model=pickle.load(open('Spam_Detector.pkl','rb'))
cv=pickle.load(open('VCC.pkl','rb'))

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify spam emails")
    st.subheader("Classification")
    user_input=st.text_area("Enter email content to classify", height=150)
    if st.button("Classify"):
        if user_input:
            data=[user_input]
            print(data)
            vec=cv.transform(data).toarray()
            result=model.predict(vec)
            if result[0]==0:
                st.success("This is not a spam email")
            else:
                st.error("This is a spam email")
        else:
            st.write("Enter content to classify")

main()