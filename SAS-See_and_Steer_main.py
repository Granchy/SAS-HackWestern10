import cv2
import dlib
import os
import numpy as np

# Initialize dlib's face detector and shape predictor (adjust the path to the shape predictor model)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/granch/Documents/hack western/Western_Hacks10/shape_predictor_68_face_landmarks.dat')

# Change lightest pixels to darkest pixels
def display_image(image):
    cv2.imshow('test image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def transform_darkest_to_white(image, filename):
  # Read the image
    threshold = 65
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Make a copy of the grayscale image
    result = grayscale_image.copy()

    # Iterate through each pixel in the grayscale image
    for i in range(grayscale_image.shape[0]):  # height
        for j in range(grayscale_image.shape[1]):  # width
            # Check if the pixel value is less than the threshold
            if grayscale_image[i, j] < threshold:
                # Change the pixel to white in the result image
                result[i, j] = 255
    

    
    # Save the result
    display_image(result)
    cv2.imwrite(filename, result)
    print(f"Image with darkest values below {threshold} swapped to white saved as '{filename}'")
    return result


# Function to get the coordinates of the left eye
def get_left_eye(frame, shape):
    left_eye_points = [shape.part(i) for i in range(36, 42)]  # Landmark indices for left eye
    left_eye_center = (int(sum([pt.x for pt in left_eye_points]) / 6),
                       int(sum([pt.y for pt in left_eye_points]) / 6))
    return left_eye_center

# Function to save the image with eyes
def save_image_with_eyes(roi, filename):
    # Save the ROI (region of interest)
    cv2.imwrite(filename, roi)
    print(f"Image with eyes saved as '{filename}'")

def grayscale_and_save(image, filename):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    cv2.imwrite(filename, gray)
    print(f"Grayscale image saved as '{filename}'")

    


def ellipse_fitting_on_roi(image, filename):
    # Fit an ellipse to the pupil region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:  # Size thresholding, adjust these values as needed
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.5 < circularity < 1.0:  # Circularity thresholding, adjust these values as needed
                valid_contours.append(contour)

    # Find the contour with the maximum area among the valid contours (likely to be the pupil)
    if valid_contours:
        max_contour = max(valid_contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(max_contour)
        ellipse_image = np.zeros_like(image)
        cv2.ellipse(ellipse_image, ellipse, (255, 255, 255), -1)

        # Calculate eccentricity
        major_axis = max(ellipse[1][0], ellipse[1][1])
        minor_axis = min(ellipse[1][0], ellipse[1][1])
        eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))

        # Add eccentricity to the fitted ellipse tuple
        fitted_ellipse = (ellipse[0], ellipse[1], ellipse[2], eccentricity)

        # Save the image with the fitted ellipse
        if ellipse_image is not None:
            cv2.imwrite(filename, ellipse_image)
            print(f"Image inside circle with fitted ellipse saved as '{filename}'")
        else:
            print("No pupil contour found.")
            
        return fitted_ellipse
    else:
        print("No valid pupil contour found.")
        return None

def detect_pupil(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=2, maxRadius=10
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            # You can choose to return the circle coordinates or perform further operations here
    return image



# Function to track the eye and display the circle
def track_eye():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Make a copy of the original frame
        original_frame = frame.copy()

        # Convert frame to grayscale for dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray)

        for face in faces:
            # Get facial landmarks
            shape = predictor(gray, face)

            # Get the center of the left eye
            left_eye_center = get_left_eye(frame, shape)

            # Define a circular mask for the left eye
            mask = np.zeros_like(frame)
            radius = 30  # Adjust the circle size as needed
            cv2.circle(mask, left_eye_center, radius, (255, 255, 255), -1)
            masked_frame = cv2.bitwise_and(frame, mask)
            

            # Draw a green circle around the detected left eye
            cv2.circle(frame, left_eye_center, radius, (0, 255, 0), 2)

            # Save the frame when 'p' is pressed
            if cv2.waitKey(1) & 0xFF == ord('p'):
                # Get the region inside the circular mask (ROI)
                roi = masked_frame[left_eye_center[1]-radius:left_eye_center[1]+radius,
                                  left_eye_center[0]-radius:left_eye_center[0]+radius]
            
        

                mask = np.zeros_like(frame)
                radius = 30  # Adjust the circle size as needed
                cv2.circle(mask, left_eye_center, radius, (255, 255, 255), -1)
                masked_frame = cv2.bitwise_and(frame, mask)

                # Use the updated fit_ellipse_on_roi function
                #fitted_ellipse_image, fitted_ellipse = fit_ellipse_on_roi(masked_frame)


                # Get the current working directory
                project_directory = os.getcwd()

                # Define the path to save the original image and the image inside the circleq
                original_image_path = os.path.join(project_directory, 'original_image.jpg')
                circle_image_path = os.path.join(project_directory, 'image_inside_circle.jpg')
                grayscale_image_path = os.path.join(project_directory, 'grayscale_image.jpg')
                ellipse_roi_image_path = os.path.join(project_directory, 'ellipse_fitted_image_roi.jpg')
                


                
           
                # Save the original frame and the region inside the circle as images
                cv2.imwrite(original_image_path, original_frame)
                save_image_with_eyes(roi, circle_image_path)
                grayscale_and_save(roi, grayscale_image_path)
                fitted_ellipse_roi = ellipse_fitting_on_roi(roi, ellipse_roi_image_path)
                



                print("Images saved.")
                print("Ellipse parameters on ROI with eccentricity:", fitted_ellipse_roi)
                cv2.imshow('Eyes Detection', roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                count  = countWhitePixels(transform_darkest_to_white(cv2.imread('/Users/granch/Documents/hack western/image_inside_circle.jpg'), 'transformed_image.jpg'))
                print(f"There are white {count} pixels in this image")
                break
            
        cv2.imshow('Eye Tracking', frame)

        # Exit the loop and close the window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def countWhitePixels(image):
     count = 0
     display_image(image)
     for i in range(image.shape[0]):  # height
        for j in range(image.shape[1]):  # width
            # Check if the pixel is a white pixel
            if image[i, j] == 255 :
                # Increment the counter by 1 when a white pixel is found
                count += 1
     return count
# Call the function to track the eye

######################################
track_eye()