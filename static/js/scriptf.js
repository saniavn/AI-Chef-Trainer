// Initialize an array to store training data
let trainingData = [];
//const checkedIngredients = Array.from(document.querySelectorAll('input[name="ingredients"]:checked')).map(el => el.value);


// Define the updateModelType function
function updateModelType() {
    const modelType = document.getElementById('model-type-dropdown') ? document.getElementById('model-type-dropdown').value : 'other';
    document.querySelectorAll('form input[name="category"]').forEach(input => {
        input.value = modelType; // Update the hidden category field in forms
    });
}

// function displayPredictionResults(data, modelIdentifier) {
//     const predictionResult = document.getElementById('prediction-result');
//
//
//     if (data && data.predictions) {
//         data.predictions.forEach(prediction => {
//             const dishElement = document.createElement('div');
//             dishElement.innerHTML = `<h3>${prediction.dish} (${prediction.probability}%)</h3>`;
//             predictionResult.appendChild(dishElement);
//         });
//     } else {
//         const noPredictionElement = document.createElement('p');
//         noPredictionElement.textContent = 'No predictions found';
//         predictionResult.appendChild(noPredictionElement);
//     }
// }

function displayPredictionResults(data, modelIdentifier) {
    const predictionResult = document.getElementById('prediction-result');

    // Create a section for this set of results
    const resultSection = document.createElement('div');
    predictionResult.appendChild(resultSection);

    // Display selected ingredients in bold
    const ingredientsInput = document.getElementById('new_ingredients').value;
    const checkedIngredients = Array.from(document.querySelectorAll('input[name="ingredients"]:checked')).map(el => el.value);
    const allIngredients = [...checkedIngredients, ...ingredientsInput.split(',').map(ingredient => ingredient.trim()).filter(ingredient => ingredient !== '')];

    const ingredientsLabel = document.createElement('div');
    ingredientsLabel.innerHTML = '<strong>Selected Ingredients:</strong> ' + allIngredients.join(', ');
    resultSection.appendChild(ingredientsLabel);

    // Create a header for predictions
    const predictionsHeader = document.createElement('div');
    predictionsHeader.innerHTML = '<strong>Recipes:</strong>';
    resultSection.appendChild(predictionsHeader);

    // List predictions
    const predictionsList = document.createElement('ul');
    if (data && data.predictions) {
        data.predictions.forEach(prediction => {
            const listItem = document.createElement('li');
            listItem.textContent = `${prediction.dish} with probability ${prediction.probability.toFixed(2)}%`;
            predictionsList.appendChild(listItem);
        });
    } else {
        const noPredictionElement = document.createElement('li');
        noPredictionElement.textContent = 'No predictions found';
        predictionsList.appendChild(noPredictionElement);
    }
    resultSection.appendChild(predictionsList);
}


document.addEventListener('DOMContentLoaded', function() {
    // Update model type selection on page load
    updateModelType();

    // Event listener for adding data to local training set
    const addToTrainingBtn = document.getElementById('add-to-training');
    if (addToTrainingBtn) {
        addToTrainingBtn.addEventListener('click', addToTraining);
    }


      // Attach event listener for Model B predictions
      const testAIButton2 = document.getElementById('test-ai-button-2');
      if (testAIButton2) {
          console.log("Attaching listener for Model B");
          testAIButton2.addEventListener('click', function(event) {
              event.preventDefault();
              console.log("Model B button clicked");
              testAIResponse2();
          });
      }


});
function displayResults(correct, userGuess, aiGuess) {
    const resultElement = document.getElementById('game-result');
    resultElement.innerHTML = '';  // Clear previous results

    const userResult = document.createElement('p');
    userResult.textContent = `Your guess: ${userGuess}`;
    const aiResult = document.createElement('p');
    aiResult.textContent = `AI's guess: ${aiGuess}`;
    const correctText = correct ? 'Correct!' : 'Not quite right.';

    const conclusion = document.createElement('h3');
    conclusion.textContent = correctText;

    resultElement.appendChild(userResult);
    resultElement.appendChild(aiResult);
    resultElement.appendChild(conclusion);
}

// // Function to get a recipe based on ingredients
// function getRecipe() {
//     // Collect ingredients from user input
//     const checkedIngredients = Array.from(document.querySelectorAll('input[name="ingredients"]:checked')).map(el => el.value);
//
//     // Send the ingredients to the Flask backend
//     fetch('/predict_recipe', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ ingredients: checkedIngredients })
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (!data || !data.recipe) {
//             throw new Error('Failed to get recipe');
//         }
//         displayRecipe(data.recipe); // Display the recipe on the webpage
//     })
//     .catch(error => {
//         console.error('Error:', error);
//         alert('Error in getting recipe: ' + error.message);
//     });
// }

// Function to display the recipe on the webpage
function displayRecipe(recipeText) {
    const recipeElement = document.getElementById('recipe-result');
    recipeElement.innerHTML = ''; // Clear previous results

    // Add the new recipe text
    const recipeContent = document.createElement('p');
    recipeContent.textContent = recipeText; // Set the recipe text
    recipeElement.appendChild(recipeContent);
}

// Add event listener for the recipe button
document.addEventListener('DOMContentLoaded', function() {
    const recipeButton = document.getElementById('get-recipe-button'); // Ensure this ID matches your HTML
    if (recipeButton) {
        recipeButton.addEventListener('click', function(event) {
            event.preventDefault(); // Prevent default form action
            getRecipe(); // Call the function to get and display the recipe
        });
    }
});
jQuery(document).ready(function() {
    $("#org").jOrgChart({
        chartElement: '#chart-container', // Optional: Specify where the chart should be placed
        dragAndDrop: true // Optional: Enable if you included jQuery UI and want drag-and-drop
    });
});


function testAIResponse2() {

  // Display the GIF container immediately
    const gifContainer = document.querySelector('.gif-container');
    gifContainer.style.display = 'block';


    // Get checked ingredients from checkboxes
    const checkedIngredients = Array.from(document.querySelectorAll('input[name="ingredients"]:checked')).map(el => el.value);

    // Get ingredients from the text field, split by commas, and trim each one
    const newIngredientsText = document.getElementById('new_ingredients').value;
    const newIngredients = newIngredientsText.split(',').map(ingredient => ingredient.trim()).filter(ingredient => ingredient !== '');

    // Combine both arrays of ingredients
     const combinedIngredients = checkedIngredients.concat(newIngredients);
//   // Ensure there are ingredients before making a request
//     if (combinedIngredients.length === 0) {
//         alert('Please enter your ingredients for recommendation.');
//         gifContainer.style.display = 'none'; // Hide GIF if no ingredients entered
//         return;
//     }
//
//
//     fetch('/predict2', { // Notice the endpoint change here
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ user_file: userFile, ingredients: combinedIngredients }) // Send combined ingredients
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (!data || !data.predictions) {
//             throw new Error('Unexpected response structure');
//         }
//     //     displayPredictionResults(data,'Model B' );
//     //     // Uncheck the checkboxes and clear the text field after displaying the prediction results
//     //     document.querySelectorAll('input[name="ingredients"]').forEach(input => {
//     //         input.checked = false;
//     //     });
//     //     document.getElementById('new_ingredients').value = ''; // Clear the text field
//     // })
//     // Use setTimeout to delay the display of results
//       setTimeout(() => {
//           displayPredictionResults(data, 'Model B');
//           // Uncheck the checkboxes and clear the text field after displaying the prediction results
//           document.querySelectorAll('input[name="ingredients"]').forEach(input => {
//               input.checked = false;
//           });
//           document.getElementById('new_ingredients').value = ''; // Clear the text field
//           gifContainer.style.display = 'none'; // Hide the GIF after displaying the results
//       }, 4000); // 4000 milliseconds = 4 seconds
//   })
//     .catch(error => {
//         console.error('Error:', error);
//         alert('Error in AI prediction 2: ' + error.message);
//     });
// }
//
// Ensure there are ingredients before making a request
    if (combinedIngredients.length === 0) {
        alert('Please enter your ingredients for recommendation.');
        gifContainer.style.display = 'none'; // Hide GIF if no ingredients entered
        return;
    }

    fetch('/predict2', { // Adjust the endpoint as necessary
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_file: userFile, ingredients: combinedIngredients }) // Send combined ingredients
    })
    .then(response => response.json())
    .then(data => {
        if (!data || !data.predictions) {
            throw new Error('Unexpected response structure');
        }

        // Use setTimeout to delay the display of results, while keeping GIF visible
        setTimeout(() => {
            displayPredictionResults(data,'Model B' );
            // Uncheck the checkboxes and clear the text field after displaying the prediction results
            document.querySelectorAll('input[name="ingredients"]').forEach(input => {
                input.checked = false;
            });
            document.getElementById('new_ingredients').value = ''; // Clear the text field
            // Optionally hide the GIF here if you want it to disappear after displaying the results
            // gifContainer.style.display = 'none';
        }, 3000); // 3000 milliseconds = 3 seconds for the delay
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error in AI prediction 2: ' + error.message);
        gifContainer.style.display = 'none'; // Ensure GIF is hidden on error
    });
}

// Function to make a prediction based on ingredients
function makePrediction() {
    const ingredients = document.getElementById('ingredients-input').value;
    console.log('Sending ingredients for prediction:', ingredients);

    fetch('/make_prediction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ingredients: ingredients.split(',').map(ingredient => ingredient.trim()) }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        console.log('Received prediction:', data);
        document.getElementById('prediction-result').textContent = 'Predicted Dish: ' + data.predicted_dish;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error making prediction: ' + error.message);
    });
}
