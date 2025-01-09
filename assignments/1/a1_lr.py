import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.linear_regression.linear_regression import lr
from PIL import Image, ImageDraw, ImageFont
import cv2
import glob
import re

def get_data(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    np.random.shuffle(data)
    x = data[:, 0].reshape(-1, 1) #setting x in data
    y = data[:, 1] #setting y in data
    return x,y

#Code provided in assignment
def make_gif(i):
    # Initialize some settings
    image_folder = "./figures/gif_images/"
    output_gif_path = f"./figures/linear_reg/linreg_k_{i}.gif"
    duration_per_frame = 50  # milliseconds

    def extract_number_from_path(path):
        match = re.search(r'image_(\d+)\.jpg', path)
        # match = re.search(r'image_\d+_(\d+)\.jpg', path)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Filename format not recognized: {path}")

    # Collect all image paths
    image_paths = glob.glob(f"{image_folder}*.jpg")
    image_paths = sorted(image_paths, key=extract_number_from_path)  # Sort the images to maintain sequence; adjust as needed

    # print(image_paths)

    # Initialize an empty list to store the images
    frames = []

    # Debugging lines
    print("Number of image paths: ", len(image_paths))
    # print("Image Paths: ", image_paths)

    # Loop through each image file to add text and append to frames
    for image_path in image_paths:
        img = Image.open(image_path)

        # Reduce the frame size by 50%
        img = img.resize((int(img.width * 0.5), int(img.height * 0.5)))

        # Create a new draw object after resizing
        draw = ImageDraw.Draw(img)

        # Text to display at top-left and bottom-right corners
        top_left_text = os.path.basename(image_path)
        bottom_right_text = "Add your text here to be displayed on Images"

        # Font settings
        font_path = "./arial.ttf"  # Replace with the path to a .ttf file on your system
        font_size = 20

        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
            print(f"Font file not found. Using default font.")

        # Draw top-left text
        draw.text((10, 10), top_left_text, font=font, fill=(255, 255, 255))

        # Calculate x, y position of the bottom-right text
        # text_width, text_height = draw.textsize(bottom_right_text, font=font)
        text_bbox = draw.textbbox((0, 0), bottom_right_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        x = img.width - text_width - 10  # 10 pixels from the right edge
        y = img.height - text_height - 10  # 10 pixels from the bottom edge

        # Draw bottom-right text
        draw.text((x, y), bottom_right_text, font=font, fill=(255, 255, 255))

        # Append the image to the frames list
        frames.append(img)

    # Debugging line after frames are added
    print("Number of frames: ", len(frames))

    # Check if frames are available before proceeding
    if frames:
        # # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
        out = cv2.VideoWriter('animation.mp4', fourcc, 20.0, (frames[0].width, frames[0].height))

        # Loop through each image frame (assuming you have the frames in 'frames' list)
        for img_pil in frames:
            # Convert PIL image to numpy array (OpenCV format)
            img_np = np.array(img_pil)

            # Convert RGB to BGR (OpenCV uses BGR instead of RGB)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Write frame to video
            out.write(img_bgr)

        # Release the VideoWriter
        out.release()

        # Save frames as an animated GIF
        frames[0].save(output_gif_path,
                       save_all=True,
                       append_images=frames[1:],
                       duration=duration_per_frame,
                       loop=0,
                       optimize=True)

        print(f"GIF saved at {output_gif_path}")

    else:
        print("No images were found to create a GIF.")

def no_regularization_3_1(file_path): #for question 3.1 (No regularization)

    #reading the data
    x,y=get_data(file_path)

    model = lr(0.01, 10000) # initializing the linear regression model with regulariation of 0

    X_train, y_train, X_val, y_val, X_test, y_test = model.split(x,y) #splitting data with a default degree of 1

    model.fit(X_train, y_train,X_val,y_val)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #mean square errors
    train_mse=model.mse(y_train,y_train_pred)
    test_mse = model.mse(y_test, y_test_pred)
    #Standard deviations
    train_sd=model.sd(y_train_pred)
    test_sd = model.sd(y_test_pred)
    #Variance
    train_var=model.var(y_train_pred)
    test_var = model.var(y_test_pred)

    print('3.1.1 Degree 1:')
    print(f"Train MSE: {train_mse}")
    print(f"Train SD: {train_sd}")
    print(f"Train var: {train_var} \n")

    print(f"Test MSE: {test_mse}")
    print(f"Test SD: {test_sd}")
    print(f"Test var: {test_var} \n")

    # #All points from dataset
    plt.scatter(X_train[:,0], y_train, color='blue', label='Train')
    plt.scatter(X_val[:,0], y_val, color='yellow', label='Validation')
    plt.scatter(X_test[:,0], y_test, color='green', label='Test')
    plt.title('Points split of data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("./figures/linear_reg/data_points.jpg")
    plt.clf()

    # # All points with model with degree 1
    # #Code for below plot was given by an LLM
    plt.scatter(X_train[:,0], y_train, color='blue', label='Train')
    plt.scatter(X_val[:,0], y_val, color='yellow', label='Validation')
    plt.scatter(X_test[:,0], y_test, color='green', label='Test')
    sorted_idx = np.argsort(X_train[:, 0])
    plt.plot(X_train[:,0][sorted_idx], y_train_pred[sorted_idx], color='black', label='Model Prediction')
    plt.title('3.1.1 Degree 1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("./figures/linear_reg/deg_1.jpg")
    plt.clf()

    best_fit=1
    min_mse=10

    for i in range(1,6):


        model = lr(0.01, 10000,make_gif=True) #change make_gif to True to generate images for gif 
        X_train, y_train, X_val, y_val, X_test, y_test = model.split(x,y,i)

        model.fit(X_train, y_train,X_val,y_val)

        #Uncomment below line to make gifs
        make_gif(i)


        y_test_pred=model.predict(X_test)
        curr_mse=model.mse(y_test,y_test_pred)
        y_train_pred = model.predict(X_train)

        #Writing the weights and biases of the best fitting model into a file
        best_fit=i if curr_mse<min_mse else best_fit
        if curr_mse<min_mse:
            with open('best_model.txt', 'w') as f:
                f.write("# Bias\n")
                f.write(f'{model.bias} \n')
                f.write("\n# Weights\n")
                f.write(f'{[w for w in model.weights]}')
        min_mse=min_mse if curr_mse>min_mse else curr_mse

        plt.scatter(X_train[:,0], y_train, color='blue', label='Train')
        plt.scatter(X_val[:,0], y_val, color='yellow', label='Validation')
        plt.scatter(X_test[:,0], y_test, color='green', label='Test')
        sorted_idx = np.argsort(X_train[:, 0])
        plt.plot(X_train[:,0][sorted_idx], y_train_pred[sorted_idx], color='black', label='Model Prediction')
        plt.title(f'Plot for k={i}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(f'./figures/linear_reg/k_{i}.jpg')
        plt.clf()

    print(f'Best fitting model degree={best_fit} with test_mse={curr_mse}')

def regularization_3_2(file_path): #for question 3.2 (with regularization)
    
    x,y=get_data(file_path)

    print(x.shape)

    for type in ["L2","L1"]: #implementing both L1 and L2 regularizations

        model = lr(0.01, 10000,25,reg_type=type)
        X_train, y_train, X_val, y_val, X_test, y_test = model.split(x,y,20)

        model.fit(X_train, y_train,X_val,y_val)

        y_train_pred = model.predict(X_train)

        y_test_pred = model.predict(X_test)

        train_mse=model.mse(y_train,y_train_pred)
        test_mse = model.mse(y_test, y_test_pred)

        train_sd=model.sd(y_train_pred)
        test_sd = model.sd(y_test_pred)

        train_var=model.var(y_train_pred)
        test_var = model.var(y_test_pred)

        print(f'{type} metrics:')
        print(f"Train MSE: {train_mse}")
        print(f"Train SD: {train_sd}")
        print(f"Train var: {train_var} \n")

        print(f"Test MSE: {test_mse}")
        print(f"Test SD: {test_sd}")
        print(f"Test var: {test_var} \n")

        #All points
        plt.scatter(X_train[:,0], y_train, color='blue', label='Train')
        plt.title('Training data points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig("./figures/linear_reg/data_points_reg.jpg")
        plt.clf()

        # All points with model with degree 1
        plt.scatter(X_train[:,0], y_train, color='blue', label='Train')
        plt.scatter(X_val[:,0], y_val, color='yellow', label='Validation')
        plt.scatter(X_test[:,0], y_test, color='green', label='Test')
        sorted_idx = np.argsort(X_train[:, 0])
        plt.plot(X_train[:,0][sorted_idx], y_train_pred[sorted_idx], color='black', label='Model Prediction')
        plt.title(f'{type} regularization result')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(f"./figures/linear_reg/reg_{type}.jpg")
        plt.clf()


def lr_main():
    file_path = r'../../data/external/linreg.csv'

    no_regularization_3_1(file_path)

    file_path = r'../../data/external/regularisation.csv'
    # regularization_3_2(file_path)
