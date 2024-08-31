
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions


# In[30]:


# Load pre-trained NASNetLarge model
model = EfficientNetB7(weights='imagenet')


# In[31]:


def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (600, 600))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = model.predict(img)
    label = decode_predictions(preds, top=1)[0][0][1]
    return label


# In[32]:


def add_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        image_path_entry.delete(0, tk.END)
        image_path_entry.insert(0, filepath)
        display_image(filepath)


# In[33]:


def display_image(filepath):
    image = Image.open(filepath)
    image = image.resize((200, 200))  # Resize image to fit the label
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo  # Keep reference to prevent garbage collection


# In[34]:


def run_classification():
    filepath = image_path_entry.get()
    if filepath:
        prediction = classify_image(filepath)
        result_label.config(text="Predicted label: " + prediction)

def close_start_page():
    start_page.destroy()

def show_instructions():
    messagebox.showinfo("Instructions", "1. Click on 'Add Image' to select an image file.\n2. Click on 'Run' to classify the image.")

# Create start page window
start_page = tk.Tk()
start_page.title("Start")
start_page.geometry("300x150")

# Create label for start page
start_label = tk.Label(start_page, text="Welcome to PetPals App", font=("Arial", 12))
start_label.pack(pady=10)

# Create Start button
start_button = tk.Button(start_page, text="Start", command=close_start_page, width=10)
start_button.pack(pady=5)

# Create Instructions button
instructions_button = tk.Button(start_page, text="Instructions", command=show_instructions, width=10)
instructions_button.pack(pady=5)

# Run the start page event loop
start_page.mainloop()

# Create main window for image classification
root = tk.Tk()
root.title("PetPals")
root.geometry("400x400")

# Create Add Image button
add_image_button = tk.Button(root, text="Add Image", command=add_image)
add_image_button.pack(pady=10)

# Create entry for image path
image_path_entry = tk.Entry(root, width=40)
image_path_entry.pack(pady=5)

# Create Label for displaying image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Create Run button
run_button = tk.Button(root, text="Run", command=run_classification)
run_button.pack(pady=10)

# Create label for displaying result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the main event loop
root.mainloop()


# In[ ]:




