import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def stream():
    st.title('Stroke Prediction')

    # Opening the dataset

    # @st.cache(allow_output_mutation=True)
    # @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("healthcare-dataset-stroke-data.csv")
        return data

    stroke_data = load_data()
    st.write("# Introduction")
    st.image("stroke.png", width=600, )

    st.write("## Stroke")
    st.write("Stroke is among major causes of death and long-term disability worldwide. "
        +"It is of great importance to predict the risk of having stroke for better prevention and early treatment."
        +" This brief report presents my attempt to develop a machine learning (ML) "
        +"model to accurately and quickly predict whether or not a person"
        +"suffered stroke based on the Kaggle stroke dataset")


    st.write("# Data information")
    st.write("## Stroke Dataset")
    st.write(stroke_data)
    st.write("## Data desciption")
    st.write(stroke_data.describe())


    # Collecting User Input
    st.sidebar.header('Stroke Prediction')

   

    st.sidebar.write('Please fill in your details below:')


    classifier_name = st.sidebar.selectbox(
        'Select a classifier',
        ('Random Forest', 'Decision Tree', 'Logistic Regression')
    )

    name = st.sidebar.text_input("Name:")

    gender_name = st.sidebar.selectbox(
        'Select Gender',
        ('Male', 'Female', 'Other')
    )

    age = st.sidebar.slider("Age:", 0, 100)
    
    hypertension = st.sidebar.selectbox(
        'Ever had a Hypertension',
        ('Yes', 'No')
    )

    heart_disease = st.sidebar.selectbox(
        'Do you have a heart disease',
        ('Yes', 'No')
    )

    marital_status = st.sidebar.selectbox(
        'Ever Married',
        ('Yes', 'No')
    )
    work_status = st.sidebar.selectbox(
        'Work Type',
        ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked')
    )
    residency_status = st.sidebar.selectbox(
        'Residency Type',
        ('Urban', 'Rural')
    )
    bmi = st.sidebar.number_input("BMI:")
    avg = st.sidebar.slider("Average Glucose Level:", 0, 300)
    smoking_status = st.sidebar.selectbox(
        'Smoking Status',
        ('formerly smoked', 'never smoked', 'smokes', 'Unknown')
    )


    # Manual Encoding
    gender_dict = {'Male': 1, 'Female': 0, 'Other': 2}
    ever_married_dict = {'No': 0, 'Yes': 1}
    work_type_dict = {'children': 0, 'Never_worked': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}
    residence_type_dict = {'Rural': 0, 'Urban': 1}
    smoking_status_dict = {'Unknown': 0, 'never smoked': 1, 'formerly smoked':2, 'smokes': 3}
    hypertension_ = {'No': 0, 'Yes': 1}
    heart_disease_ = {'No': 0, 'Yes': 1}




    gender = gender_dict[gender_name]
    marriage = ever_married_dict[marital_status]
    work = work_type_dict[work_status]
    resid = residence_type_dict[residency_status]
    smoke = smoking_status_dict[smoking_status]

    # Replacing all null values of bmi with the mean
    stroke_data['bmi'].fillna((stroke_data['bmi'].mean()), inplace=True)


    stroke_data['smoking_status'].replace(to_replace='Unknown',value=stroke_data['smoking_status'].mode()[0],inplace=True)
    stroke_data['gender'] = stroke_data['gender'].map(gender_dict)
    stroke_data['ever_married'] = stroke_data['ever_married'].map(ever_married_dict)
    stroke_data['work_type'] = stroke_data['work_type'].map(work_type_dict)
    stroke_data['Residence_type'] = stroke_data['Residence_type'].map(residence_type_dict)
    stroke_data['smoking_status'] = stroke_data['smoking_status'].map(smoking_status_dict)


    data = stroke_data
    data.drop(['id'],axis=1,inplace=True)

    X=data.drop('stroke',axis=1)
    y=data['stroke']


    # Taking a look at the distribution of class (target)
    # If the class is highly imbalanced, we have to solve this issue 
    # so that our model will not be biased towards the majority class.

    class_occur = data['stroke'].value_counts()

    st.write("### samples associated with no stroke: {}".format(class_occur[0]))
    st.write("### samples associated with stroke: {}".format(class_occur[1]))


    # Handling Imbalance
    from imblearn.over_sampling import SMOTE
    # create the  object with the desired sampling strategy.
    smote = SMOTE(sampling_strategy='minority')
    X, y = smote.fit_resample(X, y)

    # Data Splitting
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.4,random_state=40)


    def randomForestCaller():
        model=RandomForestClassifier(n_estimators=100,random_state=42)
        model=model.fit(X_train,y_train)
        pred=model.predict(X_test)
        return model, pred


    def DecisionTreeCaller():
        model=DecisionTreeClassifier()
        model=model.fit(X_train,y_train)
        pred=model.predict(X_test)
        return model, pred


    def LogisticCaller():
        model=LogisticRegression(max_iter=10000)
        model=model.fit(X_train,y_train)
        pred=mod.predict(X_test)
        return model, pred


    def get_classifier(clf_name, X_train, y_train, X_test):
        model = None
        pred = None

        if clf_name == 'Decision Tree':
            model,pred = DecisionTreeCaller()

        elif clf_name == 'Logistic':
            model,pred = LogisticCaller()
        else:
            model, pred = randomForestCaller()
        return model, pred




    #### CLASSIFICATION ####
    model, pred = get_classifier(classifier_name, X_train,y_train, X_test)




    st.write("## Confusion Matrix")
    st.write(confusion_matrix(y_test,pred))


    st.write("## Accuracy Score")
    st.write(accuracy_score(y_test,pred))


    # Calculate precision, recall, and f1 scores
    st.write("## Precision Score")
    st.write(precision_score(y_test,pred))

    st.write("## Recall Score")
    st.write(recall_score(y_test,pred))

    st.write("## F1 Score")
    st.write(f1_score(y_test,pred))

    user_data = {'Name': name,
                'Gender': gender_name,
                'Age': age,
                'Hypertension': hypertension,
                'Heart Disease': heart_disease,
                'Ever Married': marital_status,
                'Work Type': work_status,
                'Residence  Type': residency_status,
                'Avg Glucose Level': avg,
                'Bmi': bmi,
                'Smoking Status': smoking_status}


    st.header("Your details")
    st.write(pd.DataFrame(user_data, index=[0]))

    hypertension = hypertension_[hypertension]
    heart_disease = heart_disease_[heart_disease]


    prediction = model.predict([[gender, age, hypertension, heart_disease, marriage, work, resid, avg, bmi, smoke]])
    submit = st.button('Predict')

    if submit:
        st.write(prediction)
        if prediction == 0:
            st.write('## Congratulation! ', name, 'You have a low tendency for stroke')
        else:
            st.write("## We are really sorry to say ", name," but it seems like you are likely to have stroke.")



if __name__ == '__main__':
    stream()
