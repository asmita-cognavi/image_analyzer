import os
import json
from flask import Flask, request, jsonify
import google.generativeai as genai
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from imutils import face_utils
from flask import Response

# Initialize Flask app
app = Flask(__name__)

# Google GenAI setup
GOOGLE_API_KEY = 'key'  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)


def extract_json(response_text):
    """
    Extract and parse the JSON part of the response text.
    """
    try:
        # Find the JSON block in the response
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')

        # Extract the JSON substring and parse it
        if start_index != -1 and end_index != -1:
            json_string = response_text[start_index:end_index + 1]
            return json.loads(json_string)
    except json.JSONDecodeError:
        pass
    return {"error": "Failed to parse JSON from response."}

predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"{predictor_path} not found. Please ensure the file is available.")

predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    num_faces = len(faces)
    return num_faces

# 1. Brightness Check
def bright(image):
    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    brightness_threshold = (100, 250)
    brightness_result = "Good level of brightness" if brightness_threshold[0] <= brightness <= brightness_threshold[1] else "Brightness Needs Adjustment"
    return brightness_result

# 2. Sharpness (Quality) Check
def sharp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_threshold = 90
    sharpness_result = "Good Quality Image" if sharpness >= sharpness_threshold else "Blurry Image"
    return sharpness_result

# 3. Filter Check (Multiple Factors)

# Check for Saturation (high saturation may indicate filter usage)
def filters(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv_image[:, :, 1])
    saturation_threshold = 100  # High saturation may indicate a filter
    saturation_result = "Natural Image" if saturation < saturation_threshold else "May Contain Filters"

    # Check for Contrast
    contrast = np.max(image) - np.min(image)  # Difference between brightest and darkest pixel values
    contrast_threshold = 300  # High contrast might indicate a filter
    contrast_result = "Normal Contrast" if contrast < contrast_threshold else "May Contain Filters"

    # Check for Grayscale (if image is nearly grayscale, it could indicate a filter like B/W or desaturation)
    gray_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_threshold = 20  # Image with low color variance might be grayscale
    color_variance = np.std(gray_check)
    grayscale_result = "Well Colored" if color_variance > grayscale_threshold else "Grayscale Detected"
    return saturation_result,contrast_result,grayscale_result

# Check for Sepia Tone (a specific yellow-brown tint in the image)
def is_sepia(image, threshold=100):
    sepia_tones = image[:, :, 2] > image[:, :, 1]  # Red is usually stronger than green in sepia images
    sepia_ratio = np.sum(sepia_tones) / float(image.size)
    s= sepia_ratio > threshold
    if not s:
        sepia_result = "No Sepia Tone Found"
    else:
        sepia_result = "Sepia Detected"
        
    return sepia_result



# Step 4: Noise Detection (Using Your Code)

def detect_noise(image):
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Plot histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256], density=True)


    # Salt-and-Pepper Noise Check
    extreme_values = (hist[0] + hist[-1]) / 2
    if extreme_values > 0.05:  # Heuristic threshold for salt-and-pepper noise
        return "Some Noise detected"
    return "No Significant Noise detected"



def check_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define blue and white color ranges
    blue_lower = np.array([90, 50, 70])
    blue_upper = np.array([128, 255, 255])
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 55, 255])

    # Check if background is mostly blue or white
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    blue_ratio = np.sum(blue_mask) / (image.size / 3)
    white_ratio = np.sum(white_mask) / (image.size / 3)

    if (blue_ratio > white_ratio and blue_ratio > 100):
        return "Blue"
    elif (white_ratio > blue_ratio and white_ratio > 100):
        return "White"
    else:
        return "Needs Correction"


def resol(image):
    height, width = image.shape[:2]
    resolution = str(height) + " X " + str(width)
    return resolution


def check_face_centering(image):
    # Load dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    rects = detector(gray, 1)
    if len(rects) == 0:
        print("No faces detected in the image.")
        return

    # Assume a single face (e.g., for passport photo)
    rect = rects[0]

    # Calculate image center
    image_center = (image.shape[1] // 2, image.shape[0] // 2)  # (width // 2, height // 2)

    # Get facial landmarks
    shape = predictor(gray, rect)
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    # Define key points for centering checks
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    nose_tip = landmarks[30]
    lips_center = landmarks[48:54].mean(axis=0)
    jaw_midpoint = landmarks[8]  # The point at the chin

    # Define tolerances
    horizontal_tolerance = 0.05 * image.shape[1]  # 5% of image width for horizontal centering

    # Calculate horizontal distance of key points from the image center
    left_eye_dist = abs(left_eye[0] - image_center[0])
    right_eye_dist = abs(right_eye[0] - image_center[0])
    nose_dist = abs(nose_tip[0] - image_center[0])
    lips_dist = abs(lips_center[0] - image_center[0])
    jaw_dist = abs(jaw_midpoint[0] - image_center[0])

    # Calculate the horizontal midpoint between the eyes
    eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2

    # Horizontal centering check
    eye = abs(eye_midpoint_x - image_center[0])

    # Horizontal centering check for all key points
    horizontal_ok = (
        eye <= horizontal_tolerance and
        nose_dist <= horizontal_tolerance and
        lips_dist <= horizontal_tolerance and
        jaw_dist <= horizontal_tolerance
    )

    # Determine if the face is horizontally centered
    if horizontal_ok:
        h="Face is horizontally centered!"
    else:
        h="Face is not horizontally centered!"

    vertical_tolerance = 0.85 * image.shape[0]  # 8% of image height

    # Calculate the vertical center of the image
    image_vertical_center = image.shape[0] // 2

    # Get the bounding box coordinates
    bounding_box_top = rect.top()
    bounding_box_bottom = rect.bottom()

    # Apply 1% increment to the bounding box width and height
    increment_percentage = 0.01
    bounding_box_width = rect.width()
    bounding_box_height = rect.height()

    # Increase the bounding box size by 1%
    new_bounding_box_width = bounding_box_width + (bounding_box_width * increment_percentage)
    new_bounding_box_height = bounding_box_height + (bounding_box_height * increment_percentage)

    # Recalculate the new bounding box coordinates
    new_bounding_box_left = rect.left() - (new_bounding_box_width - bounding_box_width) // 2
    new_bounding_box_top = rect.top() - (new_bounding_box_height - bounding_box_height) // 2
    new_bounding_box_right = new_bounding_box_left + new_bounding_box_width
    new_bounding_box_bottom = new_bounding_box_top + new_bounding_box_height

    # Recalculate the vertical center with the expanded bounding box
    new_bounding_box_vertical_center = (new_bounding_box_top + new_bounding_box_bottom) // 2

    # Vertical centering check: if bounding box center is within vertical tolerance of image center
    vertical_ok = abs(new_bounding_box_vertical_center - image_vertical_center) <= vertical_tolerance
    verticals = abs(new_bounding_box_vertical_center - image_vertical_center)

    # Determine if the face is vertically centered
    if vertical_ok:
        v="Face is vertically centered!"
    else:
        v="Face is not vertically centered!"


    # Determine if the face is properly centered (both horizontally and vertically)
    if horizontal_ok and vertical_ok:
        c="Face is properly centered!"
    else:
        c="Face is not properly centered!"

    return h,v,c

def analyze_image_properties(image):
    # Example of calling the functions
    brightness_result = bright(image)
    sharpness_result = sharp(image)
    sepia_result=is_sepia(image)
    saturation_result, contrast_result, grayscale_result = filters(image)
    noise_result = detect_noise(image)
    background_result = check_background(image)
    resolution_result = resol(image)
    h, v, c = check_face_centering(image)
    filter_detected = "No Filters Detected"
    if saturation_result == "May Contain Filters" or contrast_result == "May Contain Filters" or grayscale_result == "Grayscale Detected" or sepia_result == "Sepia Detected":
        filter_detected = "Filters Detected"
    # Combine all results into a single dictionary
    analysis_results = {
        "resolution": resolution_result,
        "brightness": brightness_result,
        "sharpness": sharpness_result,
        "filters": filter_detected,
        "noise": noise_result,
        "background": background_result,
        "face_positioning": {
            "horizontal": h,
            "vertical": v,
            "properly_centered": c
        }
    }

    return analysis_results

def analyze_attire_and_expression(parsed_response):
    """
    Perform checks on the attire and expression values.
    """
    analysis = {}

    attire = parsed_response.get("attire")
    expression = parsed_response.get("expression")

    # Check attire value
    if attire == 0:
        analysis["attire_analysis"] = "Informal attire detected."
    elif attire == 0.5:
        analysis["attire_analysis"] = "Semi-formal attire detected."
    elif attire == 1:
        analysis["attire_analysis"] = "Formal attire detected."
    else:
        analysis["attire_analysis"] = "Attire analysis could not be determined."

    # Check expression value
    if expression == 0:
        analysis["expression_analysis"] = "Improper expression detected."
    elif expression == 1:
        analysis["expression_analysis"] = "Proper Expression detected."
    else:
        analysis["expression_analysis"] = "Expression analysis could not be determined."

    return analysis

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    # Get the uploaded image file
    image_file = request.files['image']

    # Save the image temporarily
    image_path = os.path.join("temp", image_file.filename)
    image_file.save(image_path)

    try:
        # Load the image
        image = cv2.imread(image_path)

        # 1. Call image analysis functions
        image_analysis_results = analyze_image_properties(image)

        # 2. Upload the image file to GenAI
        uploaded_file = genai.upload_file(path=image_path, display_name=image_file.filename)

        # 3. Prepare the prompt for analysis
        prompt = f"""
        Analyze the uploaded image "{uploaded_file.uri}" for a professional resume image and provide the following details:
        1. Determine if the facial expression is valid (output: 1 for neutral expression or smile, 0 otherwise for other expressions or laugh or anything loud).
        2. Identify if the attire is formal, semi-formal, or informal (output: 1 for formal which includes plain shirts or shirts with blazers, 0.5 for semi-formal where shirts have patterns like loud colors, checks or prints, 0 for informal where there are massive prints, bold colors and so on).
        Return the response in JSON format.
        Example 1:
        If a person uploads an image with neutral smile and floral shirt, the output should be:
        {{
            "expression": 1,
            "attire": 0
        }}

        Example 2:
        If a person uploads an image with laugh and mandarin collar shirt, the output should be:
        {{
            "expression": 0,
            "attire": 0
        }}

        Example 3:
        If a person uploads an image with slight smile and shirt or blazer, the output should be:
        {{
            "expression": 1,
            "attire": 1
        }}
        """

        # 4. Generate content using the Gemini model
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt, uploaded_file])

        # 5. Extract and parse the JSON response from the LLM
        parsed_response = extract_json(response.text)
        attire_and_expression_analysis = analyze_attire_and_expression(parsed_response)

        # 6. Combine both sets of results into one dictionary
        combined_results = {
            "image_analysis": image_analysis_results,
            "attire_and_expression_analysis": attire_and_expression_analysis
        }

        # Clean up temporary image file
        os.remove(image_path)
        # Convert the dictionary to a JSON string without sorting keys
        response_json = json.dumps(combined_results, sort_keys=False)

        # Return the response as a JSON response without sorting keys using Response
        return Response(response_json, content_type='application/json')


    except Exception as e:
        # Clean up temporary file on error
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    app.run(debug=True)
