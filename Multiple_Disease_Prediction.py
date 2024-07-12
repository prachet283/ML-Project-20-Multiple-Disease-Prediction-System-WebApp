# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:25:58 2024

@author: prachet
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import json
import pandas as pd


#loading the saved model of diabetes prediction
with open("Preprocessing Files/ML-Project-2-Diabetes_Prediction_Pre_Processing_Files/columns.pkl", 'rb') as f:
    all_features_diabetes_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-2-Diabetes_Prediction_Pre_Processing_Files/scaler.pkl", 'rb') as f:
    scalers_diabetes_disease = pickle.load(f)
with open("Best Features/ML-Project-2-Diabetes_Prediction_Best_Features/best_features_svc.json", 'r') as file:
    best_features_svc_diabetes_disease = json.load(file)
with open("Best Features/ML-Project-2-Diabetes_Prediction_Best_Features/best_features_lr.json", 'r') as file:
    best_features_lr_diabetes_disease = json.load(file)
with open("Best Features/ML-Project-2-Diabetes_Prediction_Best_Features/best_features_rfc.json", 'r') as file:
    best_features_rfc_diabetes_disease = json.load(file)
with open("Models/ML-Project-2-Diabetes_Prediction_Models/diabetes_disease_trained_svc_model.sav", 'rb') as f:
    loaded_model_svc_diabetes_disease = pickle.load(f)
with open("Models/ML-Project-2-Diabetes_Prediction_Models/diabetes_disease_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr_diabetes_disease = pickle.load(f)
with open("Models/ML-Project-2-Diabetes_Prediction_Models/diabetes_disease_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc_diabetes_disease = pickle.load(f)


#loading the saved model of heart disease prediction
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/columns.pkl", 'rb') as f:
    all_columns_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/cat_columns.pkl", 'rb') as f:
    cat_columns_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/encoder.pkl", 'rb') as f:
    encoder_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/encoded_columns.pkl", 'rb') as f:
    encoded_columns_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/training_columns.pkl", 'rb') as f:
    training_columns_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/scaler.pkl", 'rb') as f:
    scaler_heart_disease = pickle.load(f)
with open("Best Features/ML-Project-9-Heart_Disease_Prediction_Best_Features/best_features_xgb.json", 'r') as file:
    best_features_xgb_heart_disease = json.load(file)
with open("Best Features/ML-Project-9-Heart_Disease_Prediction_Best_Features/best_features_rfc.json", 'r') as file:
    best_features_rfc_heart_disease = json.load(file)
with open("Best Features/ML-Project-9-Heart_Disease_Prediction_Best_Features/best_features_lr.json", 'r') as file:
    best_features_lr_heart_disease = json.load(file)
with open("Models/ML-Project-9-Heart_Disease_Prediction_Models/heart_disease_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_heart_disease = pickle.load(f)
with open("Models/ML-Project-9-Heart_Disease_Prediction_Models/heart_disease_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc_heart_disease = pickle.load(f)
with open("Models/ML-Project-9-Heart_Disease_Prediction_Models/heart_disease_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr_heart_disease = pickle.load(f)


#loading the saved model of parkinson disease
with open("Preprocessing Files/ML-Project-14-Parkinson's_Disease_Prediction_Pre_Processing_Files/columns.pkl", 'rb') as f:
    all_features_parkinson_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-14-Parkinson's_Disease_Prediction_Pre_Processing_Files/scaler.pkl", 'rb') as f:
    scalers_parkinson_disease = pickle.load(f)
with open("Best Features/ML-Project-14-Parkinson's_Disease_Prediction_Best_Features/best_features_knn.json", 'r') as file:
    best_features_knn_parkinson_disease = json.load(file)
with open("Best Features/ML-Project-14-Parkinson's_Disease_Prediction_Best_Features/best_features_xgb.json", 'r') as file:
    best_features_xgb_parkinson_disease = json.load(file)
with open("Best Features/ML-Project-14-Parkinson's_Disease_Prediction_Best_Features/best_features_rfc.json", 'r') as file:
    best_features_rfc_parkinson_disease = json.load(file)
with open("Models/ML-Project-14-Parkinson's_Disease_Prediction_Models/parkinsons_disease_trained_knn_model.sav", 'rb') as f:
    loaded_model_knn_parkinson_disease = pickle.load(f)
with open("Models/ML-Project-14-Parkinson's_Disease_Prediction_Models/parkinsons_disease_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_parkinson_disease = pickle.load(f)
with open("Models/ML-Project-14-Parkinson's_Disease_Prediction_Models/parkinsons_disease_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc_parkinson_disease = pickle.load(f)


#loading the saved model of breast cancer
with open("Preprocessing Files/ML-Project-19-Breast_Cancer_Classification_Pre_Processing_Files/columns.pkl", 'rb') as f:
    all_features_breast_cancer = pickle.load(f)
with open("Preprocessing Files/ML-Project-19-Breast_Cancer_Classification_Pre_Processing_Files/scaler.pkl", 'rb') as f:
    scalers_breast_cancer = pickle.load(f)
with open("Best Features/ML-Project-19-Breast_Cancer_Classification_Best_Features/best_features_lr.json", 'r') as file:
    best_features_lr_breast_cancer = json.load(file)
with open("Best Features/ML-Project-19-Breast_Cancer_Classification_Best_Features/best_features_xgb.json", 'r') as file:
    best_features_xgb_breast_cancer = json.load(file)
with open("Best Features/ML-Project-19-Breast_Cancer_Classification_Best_Features/best_features_knn.json", 'r') as file:
    best_features_knn_breast_cancer = json.load(file)
with open("Models/ML-Project-19-Breast_Cancer_Classification_Models/parkinsons_disease_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr_breast_cancer = pickle.load(f)
with open("Models/ML-Project-19-Breast_Cancer_Classification_Models/parkinsons_disease_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_breast_cancer = pickle.load(f)
with open("Models/ML-Project-19-Breast_Cancer_Classification_Models/parkinsons_disease_trained_knn_model.sav", 'rb') as f:
    loaded_model_knn_breast_cancer = pickle.load(f)



def diabetes_prediction(input_data):

    df_diabetes_disease = pd.DataFrame([input_data], columns=all_features_diabetes_disease)

    df_diabetes_disease[all_features_diabetes_disease] = scalers_diabetes_disease.transform(df_diabetes_disease[all_features_diabetes_disease])
    
    df_best_features_svc_diabetes_disease = df_diabetes_disease[best_features_svc_diabetes_disease]
    df_best_features_lr_diabetes_disease = df_diabetes_disease[best_features_lr_diabetes_disease]
    df_best_features_rfc_diabetes_disease = df_diabetes_disease[best_features_rfc_diabetes_disease]
    
    prediction1_diabetes_disease = loaded_model_svc_diabetes_disease.predict(df_best_features_svc_diabetes_disease)
    prediction2_diabetes_disease = loaded_model_lr_diabetes_disease.predict(df_best_features_lr_diabetes_disease)
    prediction3_diabetes_disease = loaded_model_rfc_diabetes_disease.predict(df_best_features_rfc_diabetes_disease)
    
    return prediction1_diabetes_disease , prediction2_diabetes_disease, prediction3_diabetes_disease


def heart_disease_prediction(input_data):

    columns_heart_disease = all_columns_heart_disease

    df_heart_disease = pd.DataFrame([input_data], columns=columns_heart_disease)
    
    df_heart_disease[cat_columns_heart_disease] = df_heart_disease[cat_columns_heart_disease].astype('str')

    input_data_encoded_heart_disease = encoder_heart_disease.transform(df_heart_disease[cat_columns_heart_disease])

    input_data_encoded_df_heart_disease = pd.DataFrame(input_data_encoded_heart_disease, columns=encoded_columns_heart_disease)

    input_data_final_encoded_heart_disease = pd.concat([df_heart_disease.drop(cat_columns_heart_disease, axis=1).reset_index(drop=True), input_data_encoded_df_heart_disease], axis=1)

    input_data_scaled_heart_disease = scaler_heart_disease.transform(input_data_final_encoded_heart_disease)

    input_data_df_heart_disease = pd.DataFrame(input_data_scaled_heart_disease, columns=training_columns_heart_disease)

    df_best_features_xgb_heart_disease = input_data_df_heart_disease[best_features_xgb_heart_disease]
    df_best_features_rfc_heart_disease = input_data_df_heart_disease[best_features_rfc_heart_disease]
    df_best_features_lr_heart_disease = input_data_df_heart_disease[best_features_lr_heart_disease]

    prediction1_heart_disease = loaded_model_xgb_heart_disease.predict(df_best_features_xgb_heart_disease)
    prediction2_heart_disease = loaded_model_rfc_heart_disease.predict(df_best_features_rfc_heart_disease)
    prediction3_heart_disease = loaded_model_lr_heart_disease.predict(df_best_features_lr_heart_disease)
    
    return prediction1_heart_disease , prediction2_heart_disease, prediction3_heart_disease


def parkinson_disease_prediction(input_data):

    df_parkinson_disease = pd.DataFrame([input_data], columns=all_features_parkinson_disease)

    df_parkinson_disease[all_features_parkinson_disease] = scalers_parkinson_disease.transform(df_parkinson_disease[all_features_parkinson_disease])
    
    df_best_features_knn_parkinson_disease = df_parkinson_disease[best_features_knn_parkinson_disease]
    df_best_features_xgb_parkinson_disease = df_parkinson_disease[best_features_xgb_parkinson_disease]
    df_best_features_rfc_parkinson_disease = df_parkinson_disease[best_features_rfc_parkinson_disease]
    
    prediction1_parkinson_disease = loaded_model_knn_parkinson_disease.predict(df_best_features_knn_parkinson_disease)
    prediction2_parkinson_disease = loaded_model_xgb_parkinson_disease.predict(df_best_features_xgb_parkinson_disease)
    prediction3_parkinson_disease = loaded_model_rfc_parkinson_disease.predict(df_best_features_rfc_parkinson_disease)
    
    
    return prediction1_parkinson_disease , prediction2_parkinson_disease, prediction3_parkinson_disease


def breast_cancer_prediction(input_data):

    df_breast_cancer = pd.DataFrame([input_data], columns=all_features_breast_cancer)

    df_breast_cancer[all_features_breast_cancer] = scalers_breast_cancer.transform(df_breast_cancer[all_features_breast_cancer])
    
    df_best_features_lr_breast_cancer = df_breast_cancer[best_features_lr_breast_cancer]
    df_best_features_xgb_breast_cancer = df_breast_cancer[best_features_xgb_breast_cancer]
    df_best_features_knn_breast_cancer = df_breast_cancer[best_features_knn_breast_cancer]
    
    prediction1_breast_cancer = loaded_model_lr_breast_cancer.predict(df_best_features_lr_breast_cancer)
    prediction2_breast_cancer = loaded_model_xgb_breast_cancer.predict(df_best_features_xgb_breast_cancer)
    prediction3_breast_cancer = loaded_model_knn_breast_cancer.predict(df_best_features_knn_breast_cancer)
    
    return prediction1_breast_cancer , prediction2_breast_cancer, prediction3_breast_cancer




def main():
    # sidebar for navigate

    with st.sidebar:
    
        selected = option_menu('Multiple Disease Prediction System using ML',
                           
                            ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinson Disease Prediction',
                            'Breast Cancer Prediction'],
                           
                           icons = ['capsule','activity','person','virus'],
                           
                           default_index = 0)

    # Diabetes Prediction Page
    if( selected == 'Diabetes Prediction'):
        #giving a title
        st.title('Diabetes Prediction using ML')
        
        #getting input data from user

        Pregnancies = st.number_input("Number of Pregnancies",format="%.0f")
        Glucose = st.number_input("Glucose Level",format="%.2f")
        BloodPressure = st.number_input("BloodPressure volume",format="%.2f")
        SkinThickness = st.number_input("SkinThickness value",format="%.2f")
        Insulin = st.number_input("Insulin level",format="%.2f")
        BMI = st.number_input("BMI value",format="%.2f")
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction value",format="%.3f")
        Age = st.number_input("Age of the person",format="%.0f")

        

        # code for prediction
        diabetes_diagnosis_svc = ''
        diabetes_diagnosis_lr = ''
        diabetes_diagnosis_rfc = ''
        
        diabetes_diagnosis_svc,diabetes_diagnosis_lr,diabetes_diagnosis_rfc = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
        #creating a button for Prediction
        if st.button("Predict Diabetes"):
            if(diabetes_diagnosis_rfc[0]==0):
                prediction = 'The person is not diabetic' 
            else:
                prediction = 'The person is diabetic'
            st.write(f"Prediction: {prediction}")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Diabetes with Random Forest Classifier"):
                if(diabetes_diagnosis_rfc[0]==0):
                    prediction = 'The person is not diabetic' 
                else:
                    prediction = 'The person is diabetic'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Diabetes with Logistic Regression Model"):
                if(diabetes_diagnosis_lr[0]==0):
                    prediction = 'The person is not diabetic' 
                else:
                    prediction = 'The person is diabetic'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Diabetes with Support Vector Classifier"):
                if(diabetes_diagnosis_svc[0]==0):
                    prediction = 'The person is not diabetic' 
                else:
                    prediction = 'The person is diabetic'
                st.write(f"Prediction: {prediction}") 
    
 
    # Heart Disease Prediction Page
    if( selected == 'Heart Disease Prediction'):
        #giving a title
        st.title('Heart Disease Prediction using ML')
        
        #getting input data from user
            
        col1 , col2 , col3 = st.columns(3)

        with col1:
            age = st.number_input("Age in years",format="%.0f")
        with col2:
            option1 = st.selectbox('Gender',('Male', 'Female')) 
            sex = 0 if option1 == 'Female' else 1
        with col3:
            option2 = st.selectbox('Chest Pain type',('0','1','2','3'))
            if option2 == '0':
                chest_pain = 0
            elif option2 == '1':
                chest_pain = 1
            elif option2 == '2':
                chest_pain = 2
            else:
                chest_pain = 3
        with col1:
            resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)")
        with col2:
            serum_cholestoral = st.number_input("Serum Cholestoral in mg/dl")
        with col3:
            option3 = st.selectbox('Fasting Blood Sugar',('True', 'False')) 
            fasting_blood_sugar = 0 if option3 == 'False' else 1
        with col1:
            option4 = st.selectbox('Resting ECG Results',('0','1','2','3'))
            if option4 == '0':
                resting_ecg = 0
            elif option4 == '1':
                resting_ecg = 1
            elif option4 == '2':
                resting_ecg = 2
        with col2:
            max_heart_achieved = st.number_input("Maximum Heart Rate Achieved")
        with col3:
            option5 = st.selectbox('Exercise Induced Angina',('Yes', 'No')) 
            exercise_induced_angina = 0 if option5 == 'No' else 1
        with col1:
            oldpeak = st.number_input("Oldpeak (ST depression induced by exercise relative to rest)")
        with col2:
            option6 = st.selectbox('The slope of the peak exercise ST segment',('0','1','2'))
            if option6 == '0':
                slope_of_peak_exercise = 0
            elif option6 == '1':
                slope_of_peak_exercise = 1
            elif option6 == '2':
                slope_of_peak_exercise = 2
        with col3:
            option7 = st.selectbox('The slope of the peak exercise ST segment',('0','1','2','3','4'))
            if option7 == '0':
                number_of_major_vessels = 0
            elif option7 == '1':
                number_of_major_vessels = 1
            elif option7 == '2':
                number_of_major_vessels = 2
            elif option7 == '3':
                number_of_major_vessels = 3
            else:
                number_of_major_vessels = 4

        with col1:
            option7 = st.selectbox('Thal',('None','Normal','Fixed defect','Reversable defect'))
            if option7 == 'None':
                thal = 0
            elif option7 == 'Normal':
                thal = 1
            elif option7 == 'Fixed defect':
                thal = 2
            elif option7 == 'Reversable defect':
                thal = 3
        
        # code for prediction
        heart_disease_diagnosis_xgb = ''
        heart_disease_diagnosis_rfc = ''
        heart_disease_diagnosis_lr = ''
        heart_disease_diagnosis_xgb,heart_disease_diagnosis_rfc,heart_disease_diagnosis_lr =heart_disease_prediction([age,sex,chest_pain,
                                                resting_bp,serum_cholestoral,fasting_blood_sugar,
                                                resting_ecg,max_heart_achieved,exercise_induced_angina,
                                                oldpeak,slope_of_peak_exercise,
                                                number_of_major_vessels,thal])
        
        
        #creating a button for Prediction
        if st.button("Predict Heart Disease"):
            if(heart_disease_diagnosis_xgb[0]==0):
                prediction = 'The Person does not have any Heart Disease' 
            else:
                prediction = 'The Person have any Heart Disease'
            st.write(f"Prediction: {prediction}")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Heart Disease with XG Boost Classifier"):
                if(heart_disease_diagnosis_xgb[0]==0):
                    prediction = 'The Person does not have any Heart Disease' 
                else:
                    prediction = 'The Person have any Heart Disease'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Heart Disease with Random Forest Classifier"):
                if(heart_disease_diagnosis_rfc[0]==0):
                    prediction = 'The Person does not have any Heart Disease' 
                else:
                    prediction = 'The Person have any Heart Disease'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Heart Disease with Logistics Regression"):
                if(heart_disease_diagnosis_lr[0]==0):
                    prediction = 'The Person does not have any Heart Disease' 
                else:
                    prediction = 'The Person have any Heart Disease'
                st.write(f"Prediction: {prediction}")

    # Parkinson Disease Prediction Page
    if( selected == 'Parkinson Disease Prediction'):
        
        #page title
        st.title('Parkinson Disease Prediction using ML')

        col1 , col2 , col3 = st.columns(3)

        with col1:
            Fo = st.number_input("MDVP_Fo(Hz)",format="%.6f")
        with col2:
            Fhi = st.number_input("MDVP_Fhi(Hz)",format="%.6f")
        with col3:
            Flo = st.number_input("MDVP_Flo(Hz)",format="%.6f")
        with col1:
            Jitter_per = st.number_input("MDVP_Jitter(%)",format="%.6f")
        with col2:
            Jitter_Abs = st.number_input("MDVP_Jitter(Abs)",format="%.6f")
        with col3:
            RAP = st.number_input("MDVP_RAP",format="%.6f")
        with col1:
            PPQ = st.number_input("MDVP_PPQ",format="%.6f")
        with col2:
            Jitter_DDP = st.number_input("Jitter_DDP",format="%.6f")
        with col3:
            Shimmer = st.number_input("MDVP_Shimmer",format="%.6f")
        with col1:
            Shimmer_dB = st.number_input("MDVP_Shimmer(dB)",format="%.6f")
        with col2:
            Shimmer_APQ3 = st.number_input("Shimmer_APQ3",format="%.6f")
        with col3:
            Shimmer_APQ5  = st.number_input("Shimmer_APQ5",format="%.6f")
        with col1:
            APQ = st.number_input("MDVP_APQ",format="%.6f")
        with col2:
            Shimmer_DDA  = st.number_input("Shimmer_DDA",format="%.6f")
        with col3:
            NHR = st.number_input("NHR",format="%.6f")
        with col1:
            HNR = st.number_input("HNR",format="%.6f")
        with col2:
            RPDE = st.number_input("RPDE",format="%.6f")
        with col3:
            DFA = st.number_input("DFA",format="%.6f")
        with col1:
            spread1 = st.number_input("spread1",format="%.6f")
        with col2:
            spread2 = st.number_input("spread2",format="%.6f")
        with col3:
            D2 = st.number_input("D2",format="%.6f")
        with col1:
            PPE = st.number_input("PPE",format="%.6f")

        # code for prediction
        parkinson_isease_diagnosis_knn = ''
        parkinson_isease_diagnosis_xgb = ''
        parkinson_isease_diagnosis_rfc = ''
        parkinson_isease_diagnosis_knn,parkinson_isease_diagnosis_xgb,parkinson_isease_diagnosis_rfc = parkinson_disease_prediction([Fo,Fhi,Flo,Jitter_per,Jitter_Abs,RAP,PPQ,Jitter_DDP,Shimmer,Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])
        
        #creating a button for Prediction
        if st.button("Predict Parkinson Disease"):
            if(parkinson_isease_diagnosis_knn[0]==0):
                prediction = 'The Person does not have Parkinson Disease' 
            else:
                prediction = 'The Person have Parkinson Disease'
            st.write(f"Prediction: {prediction}")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Breast Cancer with K Neighbors Classifier"):
                if(parkinson_isease_diagnosis_knn[0]==0):
                    prediction = 'The Person does not have Parkinson Disease' 
                else:
                    prediction = 'The Person have Parkinson Disease'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Breast Cancer with Random Forest Classifier"):
                if(parkinson_isease_diagnosis_rfc[0]==0):
                    prediction = 'The Person does not have Parkinson Disease' 
                else:
                    prediction = 'The Person have Parkinson Disease'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Breast Cancer with XG Boost Classifier"):
                if(parkinson_isease_diagnosis_xgb[0]==0):
                    prediction = 'The Person does not have Parkinson Disease' 
                else:
                    prediction = 'The Person have Parkinson Disease'
                st.write(f"Prediction: {prediction}")
        
    
    # Breast Cancer Prediction Page
    if( selected == 'Breast Cancer Prediction'):
        #page title
        st.title('Breast Cancer Prediction using ML')

        col1 , col2 , col3 = st.columns(3)

        with col1:
            mean_radius = st.number_input("mean radius",format="%.6f")
        with col2:
            mean_texture = st.number_input("mean texture",format="%.6f")
        with col3:
            mean_perimeter = st.number_input("mean_perimeter",format="%.6f")
        with col1:
            mean_area = st.number_input("mean_area",format="%.6f")
        with col2:
            mean_smoothness = st.number_input("mean_smoothness",format="%.6f")
        with col3:
            mean_compactness = st.number_input("mean_compactness",format="%.6f")
        with col1:
            mean_concavity = st.number_input("mean_concavity",format="%.6f")
        with col2:
            mean_concave_points = st.number_input("mean_concavepoints",format="%.6f")
        with col3:
            mean_symmetry = st.number_input("mean_symmetry",format="%.6f")
        with col1:
            mean_fractal_dimension = st.number_input("mean_fractal_dim",format="%.6f")
        with col2:
            radius_error = st.number_input("radius_error",format="%.6f")
        with col3:
            texture_error  = st.number_input("texture_error",format="%.6f")
        with col1:
            perimeter_error = st.number_input("perimeter_error",format="%.6f")
        with col2:
            area_error  = st.number_input("area_error",format="%.6f")
        with col3:
            smoothness_error = st.number_input("smoothness_error",format="%.6f")
        with col1:
            compactness_error = st.number_input("compactness_error",format="%.6f")
        with col2:
            concavity_error = st.number_input("concavity_error",format="%.6f")
        with col3:
            concave_points_error  = st.number_input("concave_points_error",format="%.6f")
        with col1:
            symmetry_error = st.number_input("symmetry_error",format="%.6f")
        with col2:
            fractal_dimension_error = st.number_input("fractal_dim_error",format="%.6f")
        with col3:
            worst_radius = st.number_input("worst_radius",format="%.6f")
        with col1:
            worst_texture = st.number_input("worst_texture",format="%.6f")
        with col2:
            worst_perimeter = st.number_input("worst_perimeter",format="%.6f")
        with col3:
            worst_area  = st.number_input("worst_area",format="%.6f")
        with col1:
            worst_smoothness = st.number_input("worst_smoothness",format="%.6f")
        with col2:
            worst_compactness = st.number_input("worst_compactness",format="%.6f")
        with col3:
            worst_concavity = st.number_input("worst_concavity",format="%.6f")
        with col1:
            worst_concave_points = st.number_input("worst_concavepoints",format="%.6f")
        with col2:
            worst_symmetry = st.number_input("worst_symmetry",format="%.6f")
        with col3:
            worst_fractal_dimension = st.number_input("worst_fractal_dim",format="%.6f")

        # code for prediction
        breast_cancer_diagnosis_lr = ''
        breast_cancer_diagnosis_knn = ''
        breast_cancer_diagnosis_xgb = ''
        
        breast_cancer_diagnosis_lr,breast_cancer_diagnosis_knn,breast_cancer_diagnosis_xgb = breast_cancer_prediction([mean_radius,mean_texture,mean_perimeter,
                                             mean_area,mean_smoothness,mean_compactness,
                                             mean_concavity,mean_concave_points,mean_symmetry,
                                             mean_fractal_dimension,radius_error,
                                             texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension])
        #creating a button for Prediction
        if st.button("Predict Breast Cancer"):
            if(breast_cancer_diagnosis_xgb[0]==0):
                prediction = 'The Breast Cancer is Malignant' 
            else:
                prediction = 'The Breast Cancer is Benign'
            st.write(f"Prediction: {prediction}")
        
        if st.checkbox("Show Advanced Options"):
            if st.button("Predict Breast Cancer with XG Boost Classifier"):
                if(breast_cancer_diagnosis_xgb[0]==0):
                    prediction = 'The Breast Cancer is Malignant' 
                else:
                    prediction = 'The Breast Cancer is Benign'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Breast Cancer with Logistic Regression Model"):
                if(breast_cancer_diagnosis_lr[0]==0):
                    prediction = 'The Breast Cancer is Malignant' 
                else:
                    prediction = 'The Breast Cancer is Benign'
                st.write(f"Prediction: {prediction}")
            if st.button("Predict Breast Cancer with K Neighbors Classifier"):
                if(breast_cancer_diagnosis_knn[0]==0):
                    prediction = 'The Breast Cancer is Malignant' 
                else:
                    prediction = 'The Breast Cancer is Benign'
                st.write(f"Prediction: {prediction}")


    
if __name__ == '__main__':
    main()





