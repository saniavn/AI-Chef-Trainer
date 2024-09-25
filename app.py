from flask import Flask, request, jsonify, render_template, redirect, url_for
from joblib import load
import os
import json
import numpy as np
from sklearn.preprocessing import normalize
from flask import jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

import os
import glob
from flask import Flask, request, jsonify, send_file, redirect, url_for, render_template
from flask import jsonify, request
app = Flask(__name__, static_url_path='/static')
print("Flask app root path:", app.root_path)




new_text_model = load('text_classification_model.joblib')
#new_text_model = load('models/text_classification_model.joblib')
tfidf_vectorizer = load('models/tfidf_vectorizer.joblib')



# Load your dataset
df = pd.read_csv('mergedata.csv')

@app.route('/')
def home():
    return render_template('student_info.html')


@app.route('/supplement')
def supplement():
    return render_template('supplement.html')



@app.route('/submit_student_info', methods=['POST'])
def submit_student_info():
    # Extract and sanitize user input
    student_name = request.form.get('student_name', 'default').replace(' ', '_')
    student_age = request.form.get('student_age', 'default')
    student_grade = request.form.get('student_grade', 'default')
    user_file = f"{student_name}_{student_age}_{student_grade}.json"
    print(f"Received - Name: {student_name}, Age: {student_age}, Grade: {student_grade}")

    # Define the path for the user's file within the 'user_data' directory
    user_data_path = os.path.join(app.root_path, 'user_data', user_file)

    # Data to be written to the file
    user_data = {'name': student_name, 'age': student_age, 'grade': student_grade, 'interactions': []}

    # Write the data to a JSON file
    with open(user_data_path, 'w') as f:
        json.dump(user_data, f)

    # Redirect to another page after saving the data
    return redirect(url_for('ingredient_interaction', user_file=user_file))

@app.route('/ingredient_interaction', methods=['GET', 'POST'])
def ingredient_interaction():
    user_file = request.args.get('user_file', 'default_user.json')
    print("Received user file for ingredient interaction:", user_file)
    return render_template('ingredient_interaction.html', user_file=user_file)



#####Worked!!! My dataset on naiive bays,
@app.route('/predict2', methods=['POST'])
def predict2():
    data = request.get_json()
    user_file = data.get('user_file')
    user_ingredients = data.get('ingredients', [])

    if not user_ingredients:
        return jsonify({'error': 'Missing required data: ingredients'}), 400

    # Combine the user ingredients into a single string
    user_input = ' '.join(user_ingredients)

    # Perform the prediction directly with the pipeline
    predictions = new_text_model.predict_proba([user_input])
    class_labels = new_text_model.classes_
    top_dishes = sorted(zip(class_labels, predictions[0]), key=lambda x: x[1], reverse=True)[:3]

    # Normalize probabilities
    total_probability = sum(prob for _, prob in top_dishes)
    normalized_top_dishes = [(dish, prob / total_probability * 100) for dish, prob in top_dishes]

    # Prepare the response
    response_data = {
        'predictions': [{
            'dish': dish,
            'probability': round(prob, 2)
        } for dish, prob in normalized_top_dishes],
        'model': 'Model B'
    }

    # Update user data file as before
    user_data_path = os.path.join(app.root_path, 'user_data', user_file)
    try:
        if os.path.exists(user_data_path):
            with open(user_data_path, 'r+') as f:
                user_data = json.load(f)
                user_data['interactions'].append({
                    'ingredients': user_ingredients,
                    'predictions': response_data['predictions'],
                    'model_used': response_data['model']
                })
                f.seek(0)
                json.dump(user_data, f)
                f.truncate()
    except Exception as e:
        return jsonify({'error': f'Failed to update user file: {e}'}), 500

    return jsonify(response_data)



#
# @app.route('/classification_game')
# def classification_game():
#     return render_template('classification_game.html')  # Ensure this HTML exists

@app.route('/classification_game1')
def classification_game1():
    user_file = request.args.get('user_file', 'default_user.json')
    # Now that user_file is correctly passed, it should be accessible here
    return render_template('classification_game1.html', user_file=user_file)


# @app.route('/add_data', methods=['POST'])
# def add_data():
#     data = request.get_json()
#     print("Received data for addition:", data)
#     ingredients = data['ingredients']
#     dish_name = data['dish']
#     user_file = data['userFile']
#
#     # Path to the CSV file where data is appended
#     dataset_path = os.path.join(app.root_path, 'data', 'mergedata.csv')
#
#     # Append new data
#     with open(dataset_path, 'a') as f:
#         f.write(f'\n"{dish_name}","{ingredients}"')
#
#     # Update user interaction file
#     user_data_path = os.path.join(app.root_path, 'user_data', user_file)
#     with open(user_data_path, 'r+') as f:
#         user_data = json.load(f)
#         user_data['interactions'].append({
#             'ingredients': data['ingredients'],
#             'dish': data['dish']
#         })
#         f.seek(0)
#         json.dump(user_data, f)
#         #json.dump(user_data, f, indent=4)
#         f.truncate()
#
#     return jsonify({'message': 'Data added successfully'})

@app.route('/add_data', methods=['POST'])
def add_data():
    data = request.get_json()
    ingredients = data['ingredients']
    dish_name = data['dish']
    user_file = data['userFile']

    # Convert the user JSON file name to a CSV file name
    user_csv = user_file.replace('.json', '.csv')
    user_csv_path = os.path.join(app.root_path, 'user_data', user_csv)

    # Check if the user-specific CSV exists, if not, create it from the global template
    global_template_path = os.path.join(app.root_path, 'data', 'mergedata.csv')
    if not os.path.exists(user_csv_path):
        if os.path.exists(global_template_path):
            # Copy all data from the global template
            df_global = pd.read_csv(global_template_path)
            df_global.to_csv(user_csv_path, index=False)
        else:
            # Create an empty CSV with headers if the global template is missing
            pd.DataFrame(columns=['Dish', 'Ingredients']).to_csv(user_csv_path, index=False)

    # Append new data to the user-specific CSV
    new_row = pd.DataFrame([[dish_name, ingredients]], columns=['Dish', 'Ingredients'])
    with open(user_csv_path, 'a') as f:
        new_row.to_csv(f, header=False, index=False)

    # Update user interaction file
    user_data_path = os.path.join(app.root_path, 'user_data', user_file)
    with open(user_data_path, 'r+') as f:
        user_data = json.load(f)
        user_data['interactions'].append({
            'ingredients': ingredients,
            'dish': dish_name
        })
        f.seek(0)
        json.dump(user_data, f)
        f.truncate()

    return jsonify({'message': 'Data added successfully to user-specific CSV'})


# ###training
# @app.route('/train', methods=['POST'])
# def train_model():
#     # Path to the existing dataset
#     dataset_path = os.path.join(app.root_path, 'data', 'mergedata.csv')
#     # Check if the dataset exists
#     if not os.path.exists(dataset_path):
#         return jsonify({'error': 'Dataset not found.'}), 404
#
#     try:
#
#         # Load and prepare the dataset
#         df = pd.read_csv(dataset_path)
#         df['Ingredients'] = df['Ingredients'].astype(str)  # Convert 'Ingredients' column to string type
#
#         # Drop rows with missing values
#         df.dropna(inplace=True)
#         X = df['Ingredients'].apply(lambda x: x.lower())  # Ensure consistency in casing
#         y = df['Dish']
#
#         # Split the dataset into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#         # Define and train the model
#         model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#         model.fit(X_train, y_train)
#
#         # Evaluate the model's accuracy
#         accuracy = model.score(X_test, y_test)
#         print(f"Model accuracy: {accuracy}")
#
#         # Save the trained model
#         model_path = os.path.join(app.root_path, 'models', 'food_rec_model.joblib')
#         joblib.dump(model, model_path)
#
#         # Return success message with model accuracy
#         return jsonify({'message': 'Model trained successfully', 'accuracy': accuracy})
#     except Exception as e:
#         # Log the exception and return an error message
#         print(f"Error during model training: {e}")
#         return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    # Retrieve the user-specific CSV filename
    data = request.get_json()
    user_file = data.get('userFile', 'default_user.csv')
    user_csv_filename = user_file.replace('.json', '.csv')

    # Path to the user-specific dataset
    user_dataset_path = os.path.join(app.root_path, 'user_data', user_csv_filename)

    # Check if the user-specific dataset exists
    if not os.path.exists(user_dataset_path):
        return jsonify({'error': 'User-specific dataset not found.'}), 404

    try:
        # Load and prepare the user-specific dataset
        df = pd.read_csv(user_dataset_path)
        df['Ingredients'] = df['Ingredients'].astype(str)  # Ensure all entries are treated as strings

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Data for training
        X = df['Ingredients'].apply(lambda x: x.lower())  # Convert ingredients to lower case
        y = df['Dish']  # Target variable

        # Split the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(X_train, y_train)

        # Evaluate the model's accuracy
        accuracy = model.score(X_test, y_test)
        print(f"User-specific model accuracy: {accuracy}")

        # Save the trained model to a user-specific model file
        model_path = os.path.join(app.root_path, 'models', user_csv_filename.replace('.csv', '_model.joblib'))
        joblib.dump(model, model_path)

        return jsonify({'message': 'User-specific model trained successfully', 'accuracy': accuracy})
    except Exception as e:
        print(f"Error during model training for user {user_file}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ai_suggestions')
def ai_suggestions():
    user_file = request.args.get('user_file', 'default_user.json')
    return render_template('ai_suggestions.html', user_file=user_file)


@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    data = request.get_json()
    ingredients = data.get('ingredients', [])
    user_file = data.get('userFile')  # Retrieve userFile from the request

    ingredients_str = ', '.join(ingredients).lower()
    #model_path = os.path.join(app.root_path, 'models', 'food_rec_model.joblib')
    # Dynamically determine the model path from user file
    model_filename = user_file.replace('.json', '_model.joblib')
    model_path = os.path.join(app.root_path, 'models', model_filename)

    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found.'}), 500

    model = joblib.load(model_path)
    probabilities = model.predict_proba([ingredients_str])[0]
    class_labels = model.classes_

    # Find the top 3 predictions and their indices
    top_indices = probabilities.argsort()[-3:][::-1]
    top_predictions = [{'dish': class_labels[i], 'probability': probabilities[i]} for i in top_indices]

    # Normalize the probabilities to 100
    total = sum(prob['probability'] for prob in top_predictions)
   # normalized_predictions = [{'dish': prob['dish'], 'probability': round((prob['prob`ability'] / total) * 100, 2)} for prob in top_predictions]
    # Corrected version
    normalized_predictions = [{'dish': prob['dish'], 'probability': round((prob['probability'] / total) * 100, 2)} for
                              prob in top_predictions]

    # Path to the user's JSON file
    user_data_path = os.path.join(app.root_path, 'user_data', user_file)

    try:
        # Load the user's JSON file
        with open(user_data_path, 'r+') as f:
            user_data = json.load(f)

        # Append the normalized prediction to the user's interactions
        user_data['interactions'].append({
            'ingredients': ingredients,
            'predictions': normalized_predictions
        })

        # Save the updated data back to the file
        with open(user_data_path, 'w') as f:
            json.dump(user_data, f, indent=4)

        return jsonify(normalized_predictions)
    except IOError:
        return jsonify({'error': 'User file not found or unable to save predictions.'}), 500
    except json.JSONDecodeError:
        return jsonify({'error': 'Error decoding user JSON data.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# @app.route('/reset_session', methods=['GET'])
# def reset_session():
#     try:
#         # Path to the user_data directory
#         user_data_dir = os.path.join(app.root_path, 'user_data')
#
#         # Find all JSON files in the directory
#         json_files = glob.glob(os.path.join(user_data_dir, '*.json'))
#
#         # Loop through and delete each file
#         for file_path in json_files:
#             try:
#                 os.remove(file_path)
#                 print(f"Deleted {file_path}")
#             except Exception as e:
#                 print(f"Failed to delete {file_path}: {str(e)}")
#
#         # Reset dataset logic if needed
#         dataset_path = os.path.join(app.root_path, 'data', 'mergedata.csv')
#         df = pd.read_csv(dataset_path)
#
#         if len(df) > 78:
#             df = df[:78]
#             df.to_csv(dataset_path, index=False)
#
#         return jsonify({'message': 'User session and all user data files reset successfully'})
#     except Exception as e:
#         return jsonify({'error': f'An error occurred: {str(e)}'}), 500
##work fo rone user
# @app.route('/reset_session', methods=['GET'])
# def reset_session():
#     try:
#         # Path to the user_data directory for both JSON and CSV files
#         user_data_dir = os.path.join(app.root_path, 'user_data')
#
#         # Find all JSON files in the directory
#         json_files = glob.glob(os.path.join(user_data_dir, '*.json'))
#
#         # Loop through and delete each JSON file and its corresponding CSV file
#         for json_file_path in json_files:
#             print("Attempting to delete JSON file:", json_file_path)  # Logging path
#
#             # Construct CSV file path from JSON file path
#             csv_file_name = os.path.splitext(os.path.basename(json_file_path))[0] + '.csv'
#             csv_file_path = os.path.join(user_data_dir, csv_file_name)
#             print("Attempting to delete CSV file if exists:", csv_file_path)  # Logging path
#
#             # Delete JSON file
#             if os.path.exists(json_file_path):
#                 os.remove(json_file_path)
#                 print(f"Deleted JSON file: {json_file_path}")
#             else:
#                 print(f"JSON file not found: {json_file_path}")
#
#             # Check and delete CSV file if exists
#             if os.path.exists(csv_file_path):
#                 os.remove(csv_file_path)
#                 print(f"Deleted corresponding CSV file: {csv_file_path}")
#             else:
#                 print(f"CSV file not found: {csv_file_path}")
#
#         # Reset a global dataset or perform other cleanups
#         return jsonify({'message': 'User session and all user data files reset successfully'})
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")  # Logging the error
#         return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/reset_session', methods=['GET'])
def reset_session():
    try:
        # Path to the user_data directory for both JSON and CSV files
        user_data_dir = os.path.join(app.root_path, 'user_data')

        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(user_data_dir, '*.json'))

        # Loop through and delete each JSON file and its corresponding CSV file
        for json_file_path in json_files:
            try:
                # Delete JSON file
                if os.path.exists(json_file_path):
                    os.remove(json_file_path)
                    print(f"Deleted JSON file: {json_file_path}")
                else:
                    print(f"JSON file not found: {json_file_path}")

                # Construct CSV file path from JSON file path
                csv_file_name = os.path.splitext(os.path.basename(json_file_path))[0] + '.csv'
                csv_file_path = os.path.join(user_data_dir, csv_file_name)

                # Check and delete CSV file if exists
                if os.path.exists(csv_file_path):
                    os.remove(csv_file_path)
                    print(f"Deleted corresponding CSV file: {csv_file_path}")
                else:
                    print(f"CSV file not found: {csv_file_path}")

            except Exception as e:
                print(f"Failed to delete files for {json_file_path}: {str(e)}")

        # Reset a global dataset or perform other cleanups
        return jsonify({'message': 'User session and all user data files reset successfully'})
    except Exception as e:
        print(f"An error occurred during reset: {str(e)}")  # Logging the error
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500






@app.route('/download_user_file/<filename>')
def download_user_file(filename):
    user_data_path = os.path.join(app.root_path, 'user_data', filename)
    try:
        return send_file(user_data_path, as_attachment=True, download_name=filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found.'}), 404

@app.route('/cleanup', methods=['POST'])
def cleanup():
    data = request.get_json()
    user_file = data.get('userFile')
    file_path = os.path.join(app.root_path, 'user_data', user_file)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'message': 'User file deleted successfully'}), 200
        else:
            return jsonify({'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_page_session', methods=['GET'])
def reset_page_session():
    try:
        # Path to the user_data directory for JSON files
        user_data_dir = os.path.join(app.root_path, 'user_data')
        user_json_file = 'specific_user_file_name.json'  # Make sure to dynamically set this based on user/session

        # Full path to the JSON file
        json_file_path = os.path.join(user_data_dir, user_json_file)

        # Delete the JSON file
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
            print(f"Deleted JSON file: {json_file_path}")
        else:
            print(f"JSON file not found: {json_file_path}")

        return jsonify({'message': 'User session and JSON file reset successfully'})
    except Exception as e:
        print(f"An error occurred during reset: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/compare_guess', methods=['POST'])
def compare_guess():
    data = request.get_json()
    user_guess = data.get('guess')
    ingredients = ' '.join(data.get('ingredients', []))
    # Implement model's prediction logic here for comparison
    ai_prediction = ...  # Add the prediction logic based on ingredients
    correct = user_guess == ai_prediction  # Define correctness based on your criteria
    return jsonify({'correct': correct, 'aiPrediction': ai_prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5009)
