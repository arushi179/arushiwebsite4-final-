import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import difflib
import pickle

# Read the CSV file into a DataFrame
df = pd.read_csv('20top.csv')

# Manually specify the numerical columns
numerical_columns = [ 'WEIGHT (IN KGS)','HEIGHT (IN CMS)','BMI', 'WAIST_CMS','RBS','WHR', 'SBP']

# Replace with your actual numerical column names
categorical_columns = [col for col in df.columns if col not in numerical_columns]

# Separate the DataFrame into numerical and categorical DataFrames
numerical_df = df[numerical_columns]
categorical_df = df[categorical_columns]

# Count NaN values in each DataFrame
total_na_df = df.isna().sum().sum()
total_na_numerical = numerical_df.isna().sum().sum()
total_na_categorical = categorical_df.isna().sum().sum()

#Display the separated DataFrames
"""print("Numerical DataFrame:")
print(numerical_df)

print("\nCategorical DataFrame:")
print(categorical_df)

# Display total NaN counts
print("\nTotal NaN in original DataFrame:", total_na_df)
print("Total NaN in numerical DataFrame:", total_na_numerical)
print("Total NaN in categorical DataFrame:", total_na_categorical)
"""

# Find and print row numbers where more than 70% of the values are NaN
threshold = 0.65 * len(df.columns)
rows_with_many_nans = df.index[df.isna().sum(axis=1) > threshold].tolist()
"""print("\nRows with more than 65% NaN values:")
print(rows_with_many_nans)
print(len(rows_with_many_nans))"""

#removed the rows from entire df
df_cleaned = df.drop(rows_with_many_nans)

"""# Display the cleaned DataFrame
print("\nCleaned DataFrame:")
print(df_cleaned)
"""

# Manually specify the numerical columns
numerical_columns = [ 'WEIGHT (IN KGS)','HEIGHT (IN CMS)','BMI', 'WAIST_CMS','RBS','WHR', 'SBP']

# Identify the categorical columns by excluding the numerical columns
categorical_columns = [col for col in df_cleaned.columns if col not in numerical_columns]

# Separate the cleaned DataFrame into numerical and categorical DataFrames
cleaned_numerical_df = df_cleaned[numerical_columns]
cleaned_categorical_df = df_cleaned[categorical_columns]

"""# Display the separated DataFrames
print("Numerical DataFrame:")
print(cleaned_numerical_df)

print("\nCategorical DataFrame:")
print(cleaned_categorical_df)"""

mode_imputer = SimpleImputer(strategy='most_frequent')

# Impute missing values in categorical columns
categorical_df_imputed = pd.DataFrame(mode_imputer.fit_transform(cleaned_categorical_df), columns=cleaned_categorical_df.columns)

# Impute missing values in numerical columns
cleaned_numerical_df = df_cleaned[numerical_columns]
not_reported_counts_numerical = cleaned_numerical_df.apply(lambda col: col.isin(['Not Reported', 'NO','NR','MA','NA']).sum())
cleaned_numerical_df.replace(['Not Reported','NR', 'NA','AN','#VALUE!','N','#DIV/0!'], np.nan, inplace=True)
cleaned_numerical_df.replace(['NO','N0'], 0, inplace=True)


na_counts_numerical = cleaned_numerical_df.isna().sum()
imputer_median = SimpleImputer(strategy='median')
cleaned_numerical_df_median = pd.DataFrame(imputer_median.fit_transform(cleaned_numerical_df), columns=numerical_df.columns)
cleaned_numerical_df_median

#combining the dataframes
df_final = pd.concat([cleaned_numerical_df_median, categorical_df_imputed], axis=1)
df_final
#print(df_final)

#save it to a csv file
df_final.to_csv('Clean_Data.csv', index=True)


#Importing all the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import difflib
import pickle

df = pd.read_csv('Clean_Data.csv')

# Manually specify the numerical columns
numerical_columns = [ 'WEIGHT (IN KGS)','HEIGHT (IN CMS)','BMI', 'WAIST_CMS','RBS','WHR', 'SBP']

# Identify the categorical columns by excluding the numerical columns
categorical_columns = [col for col in df.columns if col not in numerical_columns]

# Separate the cleaned DataFrame into numerical and categorical DataFrames
cleaned_numerical_df = df[numerical_columns]
cleaned_categorical_df = df[categorical_columns]


# Apply one-hot encoding to the categorical DataFrame
categorical_df_encoded = pd.get_dummies(cleaned_categorical_df,drop_first=True).astype(int)
temp =pd.get_dummies(cleaned_categorical_df,drop_first=False).astype(int)
temp.to_csv("temp1.csv", index=None)
catgeorical_cols = ['CAUSE_MENOPAUSE_HORMONAL TREATMENT',
       'CAUSE_MENOPAUSE_HYSTERECTOMY', 'CAUSE_MENOPAUSE_NATURAL',
       'CAUSE_MENOPAUSE_NR', 'PAST_SURGERY_NR', 'PAST_SURGERY_YES',
       'STERILISATION (B/L TUBAL LIGATION)_YES',
       'NON-VEG_FREQUENCY_FORTNIGHTLY', 'NON-VEG_FREQUENCY_HALF YEARLY',
       'NON-VEG_FREQUENCY_MONTHLY', 'NON-VEG_FREQUENCY_NR',
       'NON-VEG_FREQUENCY_QUARTERLY', 'NON-VEG_FREQUENCY_THRICE WEEKLY',
       'NON-VEG_FREQUENCY_TWICE WEEKLY', 'NON-VEG_FREQUENCY_WEEKLY',
       'NON-VEG_FREQUENCY_YEARLY', 'EDUCATION_ILLITERATE',
       'EDUCATION_INTERMEDIATE', 'EDUCATION_NR', 'EDUCATION_POST GRADUATE',
       'EDUCATION_PRIMARY', 'EDUCATION_SECONDARY',
       'PHYSICAL ACTIVITY_GRADE_MODERATE', 'PHYSICAL ACTIVITY_GRADE_NIL',
       'PHYSICAL ACTIVITY_GRADE_NR', 'PHYSICAL ACTIVITY_GRADE_VIGROUS',
       'RESIDENCE_RURAL', 'RESIDENCE_URBAN', 'SES_Lower Middle', 'SES_Middle',
       'SES_Upper', 'SES_Upper Middle', 'WHO_BMI_CAT_0VERWEIGHT',
       'WHO_BMI_CAT_NORMAL', 'WHO_BMI_CAT_NR', 'WHO_BMI_CAT_OBESE CLASS I',
       'WHO_BMI_CAT_OBESE CLASS II', 'WHO_BMI_CAT_OBESE CLASS III',
       'WHO_BMI_CAT_OVERWEIGHT', 'WHO_BMI_CAT_PRE OBESE',
       'WHO_BMI_CAT_UNDERWEIGHT', 'ABD_OBESITY_#VALUE!', 'ABD_OBESITY_NO',
       'ABD_OBESITY_YES', 'DIAGNOSIS_Notdone']

catgeorical_cols_final = categorical_df_encoded[catgeorical_cols]
#print(categorical_df_encoded)
#Feature Engineering for Numerical Data
scaler = MinMaxScaler()

# Fitting the scaler and transforming the data
scaled_values = scaler.fit_transform(cleaned_numerical_df)

# Converting the scaled values back to a DataFrame
cleaned_numerical_df_median_scaled = pd.DataFrame(scaled_values, columns=cleaned_numerical_df.columns)

numerical_cols = ['WEIGHT (IN KGS)', 'HEIGHT (IN CMS)', 'BMI', 'WAIST_CMS', 'RBS', 'WHR',
       'SBP']

numerical_cols_final = cleaned_numerical_df_median_scaled[numerical_cols]

#combining the dataframes
df_final = pd.concat([numerical_cols_final, catgeorical_cols_final], axis=1)
df_final.to_csv('Feature_Eng.csv',index=False)

#Save the model
with open('scale.pkl', 'wb') as scale_file:
    pickle.dump(scaler, scale_file)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import SimpleImputer
import difflib
import pickle


#Import Dataset

df = pd.read_csv('Feature_Eng.csv')

#Define X and y
closest_match = difflib.get_close_matches('DIAGNOSIS', df.columns, n=1)

#print("Closest match for 'DIAGNOSIS':", closest_match)

# Use the closest match if found
if closest_match:
    X = df.drop(columns=[closest_match[0]])
    y = df[closest_match[0]]

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
X1_train, X1_val,y1_train,y1_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42)
#temp_df

# Initialize the classifier with class weights
best_model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42, class_weight=class_weights_dict)

# Train the model
best_model.fit(X_train, y_train)

val_predictions=best_model.predict(X1_val)
val_accuracy=accuracy_score(y1_val,val_predictions)
#print("Validation Accuracy", val_accuracy)


# Predict on the test set
test_predictions= best_model.predict(X_test)
test_accuracy=accuracy_score(y_test,test_predictions)
#print("Test Accuracy",test_accuracy)

report = classification_report(y_test, test_predictions)
#print("Classification Report:\n", report)

# Evaluate the model's performance
#accuracy = accuracy_score(y_test, y_pred)
#classification_rep = classification_report(y_test, y_pred)


#print(f"Accuracy on test set: {accuracy}")
#print("Classification report on test set:")
#print(classification_rep)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_val_score(best_model, X_train, y_train, cv=kfold, scoring='f1')

#print(f'Cross-validation f1 scores: {cv_results}')
#print(f'Mean f1: {cv_results.mean()}')
#print(f'Standard deviation: {cv_results.std()}')



feature_importances = best_model.feature_importances_

# Create a DataFrame for better visualization
feature_importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Display the DataFrame
#print("Feature importances:")
#for index, row in feature_importances_df.iterrows():
 #   print(f"{row['Feature']}: {row['Importance']:.4f}")


# Assuming 'feature_importances_df' has already been created and sorted by 'Importance'

trimmed_feature_importances_df = feature_importances_df[:-20]

# Plotting
#plt.figure(figsize=(15, 35))
#plt.barh(trimmed_feature_importances_df['Feature'], trimmed_feature_importances_df['Importance'], color='skyblue')
#plt.xlabel('Importance')
#plt.title('Feature Importances')
#plt.gca().invert_yaxis()  # Invert the y-axis to show the most important feature at the top
#plt.show()

#Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)



from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from flask_caching import Cache

app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600

cache = Cache(app, config={'CACHE_TYPE': 'simple'})
"""cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_HOST': 'localhost',
    'CACHE_REDIS_PORT': 6379,
    'CACHE_REDIS_DB': 0
})"""

# Load the scaler
with open('scale.pkl', 'rb') as scale_file:
    scaler = pickle.load(scale_file)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
@cache.cached(timeout=50)
def home():
    return render_template('index.html')

@app.route('/one(6)')
@cache.cached(timeout=50)
def one6():
    return render_template('one(6).html')


@app.route('/test-centres')
@cache.cached(timeout=50)
def testcentres():
    return render_template('test-centres.html')


@app.route('/about-us')
@cache.cached(timeout=50)
def aboutus():
    return render_template('about-us.html')

@app.route('/test-centress')
@cache.cached(timeout=50)
def testcentress():
    return render_template('test-centress.html')


@app.route('/one')
@cache.cached(timeout=50)
def one():
    return render_template('one.html')

@app.route('/two')
@cache.cached(timeout=50)
def two():
    return render_template('two.html')

@app.route('/three')
@cache.cached(timeout=50)
def three():
    return render_template('three.html')

@app.route('/four')
@cache.cached(timeout=50)
def four():
    return render_template('four.html')




def create_dummy_variables(user_input):
    temp_df = pd.read_csv("temp1.csv")
    temp_lis = ['WEIGHT (IN KGS)', 'HEIGHT (IN CMS)', 'RBS', 'WHR', 'SBP', 'WAIST_CMS', 'BMI']
    input_data = {i:0 for i in temp_df.columns if i not in temp_lis and "DIAGNOSIS" not in i}


    # Set the appropriate dummy variable to 1 based on user input
    if 'CAUSE_MENOPAUSE' in user_input:
        input_data[f'CAUSE_MENOPAUSE_{user_input["CAUSE_MENOPAUSE"]}'] = 1
    if 'PAST_SURGERY' in user_input:
        input_data[f'PAST_SURGERY_{user_input["PAST_SURGERY"]}'] = 1
    if 'STERILISATION (B/L TUBAL LIGATION)' in user_input:
        input_data[f'STERILISATION (B/L TUBAL LIGATION)_{user_input["STERILISATION (B/L TUBAL LIGATION)"]}'] = 1
    if 'NON-VEG_FREQUENCY' in user_input:
        input_data[f'NON-VEG_FREQUENCY_{user_input["NON-VEG_FREQUENCY"]}'] = 1
    if 'EDUCATION' in user_input:
        input_data[f'EDUCATION_{user_input["EDUCATION"]}'] = 1
    if 'PHYSICAL ACTIVITY_GRADE' in user_input:
        input_data[f'PHYSICAL ACTIVITY_GRADE_{user_input["PHYSICAL ACTIVITY_GRADE"]}'] = 1
    if 'RESIDENCE' in user_input:
        input_data[f'RESIDENCE_{user_input["RESIDENCE"]}'] = 1
    if 'SES' in user_input:
        input_data[f'SES_{user_input["SES"]}'] = 1
    if 'WHO_BMI_CAT' in user_input:
        input_data[f'WHO_BMI_CAT_{user_input["WHO_BMI_CAT"]}'] = 1
    if 'ABD_OBESITY' in user_input:
        input_data[f'ABD_OBESITY_{user_input["ABD_OBESITY"]}'] = 1

    # Add numeric values directly
    input_data['WEIGHT (IN KGS)'] = user_input.get('WEIGHT (IN KGS)')
    input_data['HEIGHT (IN CMS)'] = user_input.get('HEIGHT (IN CMS)')
    input_data['RBS'] = user_input.get('RBS')
    input_data['WHR'] = user_input.get('WHR')
    input_data['SBP'] = user_input.get('SBP')
    input_data['WAIST_CMS'] = user_input.get('WAIST_CMS')
    input_data['BMI'] = user_input.get('BMI')
    return input_data


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            user_input = {
                'CAUSE_MENOPAUSE': request.form['CAUSE_MENOPAUSE'],
                'PAST_SURGERY': request.form['PAST_SURGERY'],
                'WEIGHT (IN KGS)': float(request.form['WEIGHT (IN KGS)']),
                'HEIGHT (IN CMS)': float(request.form['HEIGHT (IN CMS)']),
                'STERILISATION (B/L TUBAL LIGATION)': request.form['STERILISATION (B/L TUBAL LIGATION)'],
                'NON-VEG_FREQUENCY': request.form['NON-VEG_FREQUENCY'],
                'EDUCATION': request.form['EDUCATION'],
                'PHYSICAL ACTIVITY_GRADE': request.form['PHYSICAL ACTIVITY_GRADE'],
                'RBS': float(request.form['RBS']),
                'WHR': float(request.form['WHR']),
                'RESIDENCE': request.form['RESIDENCE'],
                'SES': request.form['SES'],
                'SBP': float(request.form['SBP']),
                'WHO_BMI_CAT': request.form['WHO_BMI_CAT'],
                'WAIST_CMS': float(request.form['WAIST_CMS']),
                'ABD_OBESITY': request.form['ABD_OBESITY'],
                'BMI': float(request.form['BMI'])
            }

            # Create dummy variables
            input_data = create_dummy_variables(user_input)



            # Optionally, convert input_data to DataFrame or use it directly for prediction
            input_df = pd.DataFrame([input_data])

            # Identify numerical and categorical columns
            numerical_cols = ['WEIGHT (IN KGS)', 'HEIGHT (IN CMS)', 'BMI', 'WAIST_CMS', 'RBS', 'WHR', 'SBP']
            categorical_cols = [col for col in input_df.columns if col not in numerical_cols]

            # Scale numerical columns
            numerical_data = input_df[numerical_cols]
            scaled_numerical_features = scaler.transform(numerical_data)
            scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=numerical_cols)

            # Concatenate scaled numerical features and categorical columns
            final_df = pd.concat([scaled_numerical_df, input_df[categorical_cols]], axis=1)



            # Make predictions using the loaded model

            temp_df = pd.read_csv("Feature_Eng.csv",usecols=lambda column: column != 'DIAGNOSIS_Notdone')
            final_df = final_df[temp_df.columns]
            prediction = model.predict(final_df)

            probability_scores = model.predict_proba(final_df)
          
            probability_scores=[j for elem in probability_scores for j in elem]
            for i in range(len(probability_scores)):
                if i==0:
                    prob_0=probability_scores[i]
                    prob_0 = round((prob_0*100),2)
                    #print( prob_1)
                else:
                    prob_1 = probability_scores[i]
                    prob_1= round((prob_1*100),2)
                    # print(prob_2)


            #results['Predicted_Class']=best_model.predict(X_test)
            #results['Actual_Class']=y_test.reset_index(drop=True)
            #print("probability scores",results.head)


            if prediction == 0:
                result = f"Based on your assessment, there is a {prob_0}% estimated risk of Breast Cancer. \n \n To ensure your health and well-being, we will schedule an appointment for you at the earliest possible date.\n\n\n\n\nआपके मूल्यांकन के आधार पर, स्तन कैंसर का {prob_0}% अनुमानित खतरा है| आपके स्वास्थ्य और भले के लिए, हम आपके लिए संभाविततम तारीख पर निर्धारित करेंगे।"
            elif prediction == 1:
                result = f"You are classified in the NO risk category for Breast Cancer. \n \n There is a {prob_1}% possibility that you do not have Breast Cancer. \n \n Please continue regular follow-ups with your physician during your next visit.\n\n\n\n\nआपको स्तन कैंसर के नहीं होने की NO खतरे वाली श्रेणी में वर्गीकृत किया गया है। \n \n इसके अनुमान है कि आपके पास स्तन कैंसर नहीं है। \n \n अपने अगले आगंतुकी मुलाकात में अपने चिकित्सक के साथ नियमित अनुसरण जारी रखें।"

            # Assuming you render a result page with prediction results
            # Print or return the processed input data (for debugging or further processing)
            final_df.to_csv('final_df.csv', index=False)


            # Render the template with the prediction string
            return render_template('result(6).html', prediction=result)


        except Exception as e:

            return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
