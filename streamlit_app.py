import streamlit as st
from streamlit_cropper import st_cropper
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import sleep
import requests
import random
from tqdm import tqdm
import tensorflow as tf
from keras import models, layers
import shutil

# Helpers
def create_dataset(size, train_count, val_count, train_progress=None, val_progress=None, log_callback=None):
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)

    # Train dirs
    train_dir = dataset_dir / "train"
    train_hr_dir = train_dir / "HR"
    train_lr_dir = train_dir / "LR"
    train_hr_dir.mkdir(parents=True, exist_ok=True)
    train_lr_dir.mkdir(parents=True, exist_ok=True)

    # Val dirs
    val_dir = dataset_dir / "val"
    val_hr_dir = val_dir / "HR"
    val_lr_dir = val_dir / "LR"
    val_hr_dir.mkdir(parents=True, exist_ok=True)
    val_lr_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://picsum.photos/{size}"

    # Train
    load_subset(train_count, base_url, train_hr_dir, train_lr_dir, size, progress_callback=train_progress, log_callback=log_callback)

    # Vald
    load_subset(val_count, base_url, val_hr_dir, val_lr_dir, size, progress_callback=val_progress, log_callback=log_callback)


def load_subset(count, base_url, hr_dir, lr_dir, size, progress_callback=None, log_callback=None):
    for i in range(count):
        try:
            response = requests.get(base_url, timeout=10)
            if response.status_code == 200:
                file_path = hr_dir / f"{i:04d}_HR.jpg"

                # Save HR image
                with open(file_path, "wb") as f:
                    f.write(response.content)

                # Save LR image
                pixelate(file_path, lr_dir, i, size)

        except Exception as e:
            if log_callback:
                log_callback(f"error downloading image {i+1}: {e}")

        finally:
            if progress_callback:
                progress_callback(i + 1, count)

def pixelate(img_path, lr_dir, index, size):
    hr = cv2.imread(str(img_path))  
    if hr is None:
        print(f"failed to load image: {img_path}")
        return

    # create random pix factor in range [4, 8]
    random_factor = random.randint(4, 8)

    # pixelate HR
    temp = cv2.resize(hr, (size // random_factor, size // random_factor), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(temp, (size, size), interpolation=cv2.INTER_NEAREST)
    file_path = lr_dir / f"{index:04d}_LR.jpg"

    # save LR
    cv2.imwrite(str(file_path), pixelated)
    return

# logic for converting dataset to tf.data.Dataset
def load_image_pair(lr_path, hr_path):
    # Read LR and HR images
    lr = tf.io.read_file(lr_path)
    hr = tf.io.read_file(hr_path)
    lr = tf.image.decode_jpeg(lr, channels=3)
    hr = tf.image.decode_jpeg(hr, channels=3)

    # Convert to float32 and normalize to [0, 1]
    lr = tf.image.convert_image_dtype(lr, tf.float32)
    hr = tf.image.convert_image_dtype(hr, tf.float32)

    return lr, hr

def get_dataset(lr_dir, hr_dir, batch_size=8, shuffle=True):
    lr_paths = sorted([str(p) for p in Path(lr_dir).glob("*.jpg")])
    hr_paths = sorted([str(p) for p in Path(hr_dir).glob("*.jpg")])

    dataset = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths)) # Should be <class 'str'>
    dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(lr_paths))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def clear_dataset():
    dataset_dir = Path("dataset")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

# Model functions
def create_model(optimizer='adam', loss='mean_squared_error'):
    model = models.Sequential()
    # Input layer
    model.add(layers.Input(shape=(None, None, 3)))
    # Patch extraction and representation
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same'))
    # Non-linear mapping
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same'))
    # Reconstruction
    model.add(layers.Conv2D(3, (5, 5), activation='linear', padding='same'))

    # compile and return model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# Model testing
def pixelate_image(img, size, factor=None):
    """Pixelate the input image by downsampling and upsampling using nearest neighbor."""
    if factor is None:
        factor = random.randint(4, 8)
    small_size = size // factor
    temp = cv2.resize(img, (small_size, small_size), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(temp, (size, size), interpolation=cv2.INTER_NEAREST)
    return pixelated

def crop_center(img, size):
    """Crop the center square of given size from the image."""
    h, w = img.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    cropped = img[top:top+size, left:left+size]
    return cropped

def run_test(image, model, size):
    # Load and crop image to size x size
    img = crop_center(image, size)

    # Convert BGR (cv2 default) to RGB for plotting & model input
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create pixelated LR version with random factor
    lr_img = pixelate_image(img_rgb, size)

    # Normalize input for model ([0,1] float32)
    lr_norm = lr_img.astype(np.float32) / 255.0
    lr_norm = np.expand_dims(lr_norm, axis=0)  # batch dimension

    # Run model prediction (super-resolved image)
    sr_img = model.predict(lr_norm)[0]  # remove batch dim

    # Clip and convert output to uint8 for display
    sr_img = np.clip(sr_img, 0, 1)
    sr_img = (sr_img * 255).astype(np.uint8)

    return img_rgb, lr_img, sr_img

# Custom loss
def ssim(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# App
# initialize session state variables
if "dataset_ready" not in st.session_state:
    st.session_state.dataset_ready = False
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "training_finished" not in st.session_state:
    st.session_state.training_finished = False

# load css styliing
st.markdown("""<style> 
                div.stButton button{
                width: 100%;
                }
                </style>
            """
            , unsafe_allow_html=True)

# Parameters Sidebar
st.sidebar.image("logo.png", use_container_width=True)
img_size = st.sidebar.number_input("Training Image Size", min_value=64, max_value=512, value=256, step=32)
train_count = st.sidebar.number_input("Num Training Samples", min_value=1, value=80, step=10)
val_count = st.sidebar.number_input("Num Validation Samples", min_value=1, value=20, step=10)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
batch_size = st.sidebar.selectbox("Batch Size", [4, 8, 16, 32], index=1)
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "rmsprop", "sgd"], index=0)
loss = st.sidebar.selectbox("Loss Function", ["mean_squared_error", "mean_absolute_error", 
                                               "huber_loss", "ssim"], index=0)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Output Log", "Model Performance", "Test Output", "Preview Dataset"])
with tab1:
    output_log = st.container(height=600, border=True)
    with output_log:
        st.info("Welcome to the SRCNN Demo!")
        st.markdown("""
                        For more infomation, check out my video series on YouTube:
                    
                        - [Part 1: Intro](https://youtu.be/mkUhGvbmhSI?si=FUhbOtpsg4SGqRoh)
                        - [Part 2: Theory](https://youtu.be/qZcBXYQLLyQ?si=YJzQ2swBXg-BZy1-)
                        - coming soon
                    
                        Demo Instructions:
                        1. choose parameters in the sidebar
                        2. click `Generate Dataset` to generate a training and vaidation dataset
                        3. click `Train Model` to build and train the SRCNN model
                        4. upload a custom image for the model test if desired, or use the default image
                        5. click `Test Model` to test the trained model on a sample image         
                    """)
        st.write("---")
with tab2:
    # Check that all required metrics are available
    has_loss = "train_loss" in st.session_state and "val_loss" in st.session_state
    has_acc = "train_acc" in st.session_state and "val_acc" in st.session_state

    if has_loss or has_acc:
        col1, col2 = st.columns(2)

        # Plot Loss in left column
        if has_loss:
            with col1:
                train_loss = st.session_state.train_loss
                val_loss = st.session_state.val_loss
                epochs = range(1, len(train_loss) + 1)

                fig_loss, ax1 = plt.subplots()
                ax1.set_title("Training vs Validation Loss")
                ax1.plot(epochs, train_loss, label='Train Loss', marker='o')
                ax1.plot(epochs, val_loss, label='Validation Loss', marker='o')
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.set_xticks(epochs)
                ax1.legend()
                st.pyplot(fig_loss)

        # Plot Accuracy in right column
        if has_acc:
            with col2:
                train_acc = st.session_state.train_acc
                val_acc = st.session_state.val_acc
                if any(v is not None for v in train_acc) and any(v is not None for v in val_acc):
                    epochs = range(1, len(train_acc) + 1)

                    fig_acc, ax2 = plt.subplots()
                    ax2.set_title("Training vs Validation Accuracy")
                    ax2.plot(epochs, train_acc, label='Train Accuracy', marker='o')
                    ax2.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Accuracy")
                    ax2.set_xticks(epochs)
                    ax2.legend()
                    st.pyplot(fig_acc)

    else:
        st.warning("No performance data available, train model first")

with tab3:
    if "hr_img" in st.session_state and "lr_img" in st.session_state and "sr_img" in st.session_state:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(st.session_state.hr_img, caption="High Resolution (HR)", use_container_width=True)

        with col2:
            st.image(st.session_state.lr_img, caption="Low Resolution (LR)", use_container_width=True)

        with col3:
            st.image(st.session_state.sr_img, caption="Super-Resolved (SR)", use_container_width=True)
    else:
        st.warning("No test results available, please run 'Test Model' first")
            
with tab4:
    # Dropdowns for dataset and resolution
    dataset_type = st.selectbox("Select Dataset", ["train", "val"])
    resolution_type = st.selectbox("Select Resolution", ["HR", "LR"])

    folder_path = Path("dataset") / dataset_type / resolution_type

    if not folder_path.exists():
        st.warning("No dataset has been generated, please generate the dataset before previewing")
    else:
        # List image files
        image_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])

        if not image_files:
            st.warning("No images found in the selected folder")
        else:
            # Slider for number of images per page
            images_per_page = st.slider("Images per page", min_value=4, max_value=40, value=12, step=4)

            # Slider for page number
            max_page = (len(image_files) - 1) // images_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=max_page, value=1)

            # Compute range
            start_idx = (page - 1) * images_per_page
            end_idx = start_idx + images_per_page
            files_to_display = image_files[start_idx:end_idx]

            # Display images in a 4-column grid
            cols = st.columns(4)
            for i, image_path in enumerate(files_to_display):
                # Read and convert BGR to RGB
                img = cv2.imread(str(image_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with cols[i % 4]:
                    st.image(img_rgb, caption=image_path.name, use_container_width=True)

# Control buttons
st.sidebar.markdown("---")
if st.sidebar.button("Refresh Session"):
    with output_log: 
        clear_dataset()
        st.session_state.clear() 
        st.success("Session refreshed. All state cleared.")
        sleep(1)
        st.rerun()

if st.sidebar.button("Generate Dataset"):
    with output_log:
        if not st.session_state.dataset_ready:
            st.write("\>> Creating dataset")

            # Initialize progress bars
            train_prog = st.progress(0, text=f"Downloading training set: 0%")
            val_prog = st.progress(0, text=f"Downloading validation set: 0%")

            # Progress update functions
            def update_train_progress(current, total):
                train_prog.progress(current / total, text=f"Downloading training set: {int((current / total)*100)}%")

            def update_val_progress(current, total):
                val_prog.progress(current / total, text=f"Downloading validation set: {int((current / total)*100)}%")

            def log_message(message):
                st.write(message)

            try:
                create_dataset(
                    img_size, train_count, val_count,
                    train_progress=update_train_progress,
                    val_progress=update_val_progress,
                    log_callback=log_message
                )

                st.session_state.dataset_ready = True
                st.success("Dataset ready!.")
                sleep(1)
                st.rerun()  # Refresh the app to update state

            except Exception as e:
                st.error(f"Error generating dataset: {e}")
        else:
            st.info("Dataset already generated. Skipping.")

if st.sidebar.button("Train Model"):
    with output_log:
        if st.session_state.dataset_ready:
            st.write("\>> Creating model")
            try:
                model = create_model(optimizer=optimizer, loss=ssim if loss == "ssim" else loss)
                st.session_state.model = model
                st.session_state.model_ready = True
            except Exception as e:
                st.error(f"Error creating model: {e}")
                st.stop()

            st.write("\>> Preparing datasets")
            try:
                ds_train = get_dataset("dataset/train/LR", "dataset/train/HR", batch_size=batch_size)
                ds_val = get_dataset("dataset/val/LR", "dataset/val/HR", batch_size=batch_size)
                st.session_state.ds_train = ds_train
                st.session_state.ds_val = ds_val
                st.session_state.img_size = img_size
            except Exception as e:
                st.error(f"Error loading datasets: {e}")
                st.stop()

            st.write("\>> Training model")
            try:
                train_loss_history = []
                val_loss_history = []
                train_acc_history = []
                val_acc_history = []

                for i in range(epochs):
                    history = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=0)

                    train_loss = history.history.get("loss", [None])[0]
                    val_loss = history.history.get("val_loss", [None])[0]
                    train_acc = history.history.get("accuracy", [None])[0]
                    val_acc = history.history.get("val_accuracy", [None])[0]

                    train_loss_history.append(train_loss)
                    val_loss_history.append(val_loss)
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)

                    with st.container(border=True):
                        st.success(f"Epoch {i+1}/{epochs} complete")
                        if train_acc is not None and val_acc is not None:
                            st.write(f"Training accuracy: {train_acc:.4f}")
                            st.write(f"Validation accuracy: {val_acc:.4f}")
                        else:
                            st.write("Accuracy metrics not available in history")

                st.session_state.train_loss = train_loss_history
                st.session_state.val_loss = val_loss_history
                st.session_state.train_acc = train_acc_history
                st.session_state.val_acc = val_acc_history
                st.session_state.training_finished = True

                st.success("Training complete!")
                sleep(1)
                st.rerun()  # Refresh the app to update state

            except Exception as e:
                st.error(f"Error during training: {e}")
                st.stop()

        else:
            st.warning("Please generate dataset first")

img_file = st.sidebar.file_uploader("Upload Custom Test Image", type=["jpg", "jpeg", "png"])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
if img_file:
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=(1, 1))
    
    # Manipulate cropped image at will
    st.session_state.uploaded_img = cropped_img

if st.sidebar.button("Test Model"):
    with output_log:
        if st.session_state.training_finished:
            try:
                # use custom test image if available, otherwise use default
                if "uploaded_img" in st.session_state:
                    test_image = st.session_state.uploaded_img
                else: 
                    test_image = cv2.imread("willy.JPG", cv2.IMREAD_COLOR)

                hr_test, lr_test, sr_test = run_test(
                    test_image, 
                    st.session_state.model, 
                    st.session_state.img_size
                )

                st.session_state.hr_img = hr_test
                st.session_state.lr_img = lr_test
                st.session_state.sr_img = sr_test

                st.session_state.test_ready = True
                st.success("Model testing complete!")
                sleep(1)
                st.rerun()  # Refresh the app to update state

            except Exception as e:
                st.error(f"Error during model test: {e}")
        else:
            st.warning("Please train model first")
