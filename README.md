# Food Image Classification and Nutrition Prediction

## Overview

Food Image Classification and Nutrition Prediction is a deep learning-based application that identifies food items from uploaded images and estimates their nutritional values. The system helps users understand the nutritional content of their meals by combining image classification with nutrition data from the USDA FoodData Central database.

## Objectives

* Classify food items from images using deep learning.
* Predict nutritional information such as calories, carbohydrates, proteins, and fats.
* Promote healthier dietary choices by providing quick nutrition insights.
* Reduce manual food logging through automated food recognition.

## Features

* Upload a food image for analysis.
* Automatic food image classification.
* Nutrition prediction using the USDA nutrition database.
* Displays estimated nutritional values.
* User-friendly interface for food analysis.

## Technologies Used

* Python
* TensorFlow
* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* USDA FoodData Central Dataset

## Project Workflow

1. Collect and preprocess the food image dataset.
2. Train the deep learning model for food classification.
3. Upload a food image.
4. Predict the food category.
5. Retrieve nutritional information from the USDA dataset.
6. Display the predicted nutritional values to the user.

## Repository Structure

```text
Food-Image-Classification/
├── dataset/
├── models/
├── notebooks/
├── src/
├── images/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Jestins04/food_image_classification.git
cd food_image_classification
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python app.py
```

Upload a food image and view the predicted food category along with its estimated nutritional values.

## Future Enhancements

* Improve prediction accuracy with larger datasets.
* Support multiple food items in a single image.
* Real-time mobile application integration.
* Personalized dietary recommendations.
* Barcode and meal tracking support.

## Team Project

This project was developed collaboratively as part of a college team project.

### My Contributions

* Assisted in the development and implementation of the Food Image Classification module.
* Contributed to nutrition prediction using the USDA nutrition database.
* Participated in data preprocessing, model evaluation, testing, and documentation.
* Collaborated with team members during project development and integration.

## Acknowledgements

* USDA FoodData Central
* TensorFlow
* PyTorch
* OpenCV
* Python Open Source Community

## License

This project is intended for educational and academic purposes.
