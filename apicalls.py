
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

#to make the flask app 




import random
from deepface import DeepFace
from transformers import pipeline

#  emotion classifier from hugging face 
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

# greetings for the user when they open the app 
greetings = {
    "general": [
        "Hello {name}, I hope you're having a wonderful day!",
        "Hi {name}, just wanted to drop in and say you're amazing!",
        "Good day {name}, wishing you the best!",
        "Hey {name}, stay awesome and keep shining!",
    ],
    "motivation": [
        "Hi {name}, you’ve got this! Keep pushing forward!",
        "Hey {name}, remember, every step counts. You’re doing great!",
        "Hello {name}, stay strong and keep believing in yourself!",
        "Hi {name}, you are capable of amazing things!",
    ],
    "sweetheart": [
        "Hi {name}, take care of yourself today!",
        "Hey {name}, you're so important to me.",
        "Hi {name}, sending you love and care.",
        "Hello {name}, remember, you matter.",
        "Hey {name}, hope you're feeling good today!",
    ]
}

# function to randomly generate greetings based on topic 
def generate_greeting(name, topic="general"):
    if topic in greetings:
        return random.choice(greetings[topic]).format(name=name)
    else:
        return random.choice(greetings["general"]).format(name=name)

# testing the function
name = "Rebecca"
topic = "sweetheart"  #  can change the topic to "general", "motivation", or "sweetheart"
greeting = generate_greeting(name, topic)
print(greeting)

#extracting emotions from images function 

def get_image_emotion(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'])
        dominant_emotion = result['dominant_emotion']
        return dominant_emotion, result['emotion'][dominant_emotion]  # Return emotion and confidence score
    except Exception as e:
        return None, None  # Return None if DeepFace fails

#function to extract emotions from text 
def get_text_emotion(text):
    try:
        result = emotion_classifier(text)
        emotion = result[0]['label']
        return emotion, result[0]['score']  # return emotion and confidence score
    except Exception as e:
        return None, None  # return none if the extraction fails 
    
#function to take the text and image detection (we can refine this later)
def combine_emotions(text_emotion, image_emotion):
    stress_emotions = ['anger', 'fear', 'sad', 'disgust', 'stress']
    
    if any(emotion in text_emotion for emotion in stress_emotions) or image_emotion in stress_emotions:
        return 'High Stress'
    elif text_emotion == 'neutral' and image_emotion == 'neutral':
        return 'Low Stress'
    else:
        return 'Medium Stress'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'text' not in request.json:
        return jsonify({'error': 'Missing image or text input'}), 400

    # Get text input
    text = request.json['text']
    
    # Save the image temporarily
    image = request.files['image']
    image_path = "temp_image.jpg"
    image.save(image_path)
    
    # Get emotions from text and image
    text_emotion, _ = get_text_emotion(text)
    image_emotion, _ = get_image_emotion(image_path)
    
    # Combine emotions to determine stress level
    stress_level = combine_emotions(text_emotion, image_emotion)
    
    # Clean up the temp image file
    os.remove(image_path)
    
    return jsonify({'stress_level': stress_level})

if __name__ == '__main__':
    app.run(debug=True)


