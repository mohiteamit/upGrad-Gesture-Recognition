from typing import List, Tuple
import random
import os
import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError:
    print("mediapipe module not found. Skipping...")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

def set_memorry_limit_for_tf():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPUs")
        except RuntimeError as e:
            print(e)

    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=20000)]  # Limit to 20GB
            )
            print("GPU memory limited to 20GB")
        except RuntimeError as e:
            print(e)

def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set to {seed}")

def plot_training_history(histories: List, ylim_loss: Tuple[float, float] = (0, 3)) -> None:
    """
    Plots combined training and validation loss and accuracy from multiple Keras history objects.
    The histories are combined to produce a single plot with consistent scales for comparability.

    Args:
        histories (List): A list of Keras History objects returned by model.fit().
        ylim_loss (Tuple[float, float], optional): 
            A tuple specifying the y-axis range for the loss plot. Default is (0, 3).

    Returns:
        None: The function creates and displays the plots but does not return any value.
    """
    # Initialize combined metrics
    combined_acc = []
    combined_val_acc = []
    combined_loss = []
    combined_val_loss = []

    # Combine metrics from all histories
    for history in histories:
        combined_acc.extend(history.history.get('accuracy', []))
        combined_val_acc.extend(history.history.get('val_accuracy', []))
        combined_loss.extend(history.history.get('loss', []))
        combined_val_loss.extend(history.history.get('val_loss', []))

    # Create epoch range based on combined history length
    epochs = range(1, len(combined_loss) + 1)

    # Create subplots for side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training and validation accuracy
    if combined_acc:
        axes[0].plot(epochs, combined_acc, 'b-', label='Training Accuracy')
        if combined_val_acc:
            axes[0].plot(epochs, combined_val_acc, 'r-', label='Validation Accuracy')
        axes[0].set_title('Combined Training and Validation Accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1.0)  # Fixed scale for accuracy
        axes[0].legend()

    # Plot training and validation loss
    if combined_loss:
        axes[1].plot(epochs, combined_loss, 'b-', label='Training Loss')
        if combined_val_loss:
            axes[1].plot(epochs, combined_val_loss, 'r-', label='Validation Loss')
        axes[1].set_title('Combined Training and Validation Loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].set_ylim(ylim_loss)  # Fixed scale for loss
        axes[1].legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

class MediaPipeHandProcessor:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        else:
            keypoints = np.zeros((21, 3))  # No hand detected

        return keypoints

    def process_sequence(self, sequence_frames: np.ndarray) -> np.ndarray:
        return np.array([self.process_frame(frame) for frame in sequence_frames])

class GestureDataGenerator(Sequence):
    def __init__(
        self, 
        data_path: str, 
        labels_csv: str, 
        batch_size: int, 
        image_size=(224, 224), 
        augmentations=None, 
        shuffle: bool = False, 
        load_fraction: float = 1.0,
        use_mediapipe: bool = False,
        sequence_length: int = 30,
        debug: bool = False,
        seed: int = 42,
    ):
        assert 0 < load_fraction <= 1.0, "load_fraction must be between 0 and 1"

        self.seed = seed
        np.random.seed(self.seed)

        super().__init__()
                
        self.data_path = data_path
        self.labels = self._load_labels(labels_csv)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.use_mediapipe = use_mediapipe
        self.sequence_length = sequence_length
        self.debug = debug
        self.mediapipe_processor = MediaPipeHandProcessor() if use_mediapipe else None

        # Prepare list of sequences
        self.sequence_paths = list(self.labels.keys())
        self._filter_fraction(load_fraction)
        if self.shuffle:
            np.random.shuffle(self.sequence_paths)

        # Print details of the generator
        self._print_details()

    def _load_labels(self, labels_csv):
        labels = {}
        with open(labels_csv, 'r') as f:
            for line in f:
                sequence_name, _, label = line.strip().split(';')
                labels[sequence_name] = int(label)
        return labels

    def _filter_fraction(self, load_fraction: float) -> None:
        total_sequences = len(self.sequence_paths)
        num_sequences = int(total_sequences * load_fraction)
        self.sequence_paths = self.sequence_paths[:num_sequences]

    @property
    def num_classes(self) -> int:
        return len(set(self.labels.values()))

    def __len__(self) -> int:
        return int(np.ceil(len(self.sequence_paths) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_sequences = self.sequence_paths[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self._generate_data(batch_sequences)
        return X, y

    def _generate_data(self, batch_sequences: List[str], debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []

        for seq_name in batch_sequences:
            seq_path = os.path.join(self.data_path, seq_name)
            frames = sorted([
                os.path.join(seq_path, img) for img in os.listdir(seq_path)
                if img.endswith(('.jpg', '.png'))
            ])[:self.sequence_length]

            sequence_images = []

            if self.use_mediapipe:
                # Process frames with MediaPipe for keypoints
                for frame_path in frames:
                    img = cv2.imread(frame_path)
                    resized_img = self._resize_and_normalize_frame(img, debug)
                    keypoints = self.mediapipe_processor.process_frame(resized_img)  # Use resized image for MediaPipe
                    sequence_images.append(keypoints)
                sequence_images = np.array(sequence_images, dtype=np.float32)  # (sequence_length, 21, 3)
            else:
                # Process frames normally
                for frame_path in frames:
                    img = cv2.imread(frame_path)
                    resized_img = self._resize_and_normalize_frame(img, debug)
                    sequence_images.append(resized_img)

                sequence_images = np.array(sequence_images, dtype=np.float32)  # (sequence_length, height, width, channels)

                # Apply augmentations if defined
                if self.augmentations:
                    sequence_images = self._apply_augmentations(sequence_images)

            # Append the processed sequence and corresponding label
            X.append(sequence_images)
            y.append(self.labels[seq_name])

        # Convert to numpy arrays
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        if debug:
            print(f"Generated batch X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def _apply_augmentations(self, sequence_images: np.ndarray) -> np.ndarray:
        if not self.augmentations:              # If no augmentations are provided, return as is
            return sequence_images

        augmented_sequence = []
        rng = np.random.RandomState(self.seed)  # Ensure reproducibility

        for img in sequence_images:
            augmented_img = img.copy()

            # Apply rotation
            if 'rotation' in self.augmentations:
                angle = rng.uniform(-self.augmentations['rotation'], self.augmentations['rotation'])
                h, w = augmented_img.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented_img = cv2.warpAffine(augmented_img, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Apply brightness adjustment
            if 'brightness' in self.augmentations:
                factor = rng.uniform(0.9, 1.1)
                augmented_img = np.clip(augmented_img * factor, 0, 255)

            # Apply contrast adjustment
            if 'contrast' in self.augmentations:
                factor = rng.uniform(0.9, 1.1)
                augmented_img = np.clip(128 + factor * (augmented_img - 128), 0, 255)

            # Apply blur
            if 'blur' in self.augmentations:
                augmented_img = cv2.GaussianBlur(augmented_img, (3, 3), 0)

            augmented_sequence.append(augmented_img)

        return np.array(augmented_sequence, dtype=np.float32)

    def _resize_and_normalize_frame(self, img: np.ndarray, debug: bool = False) -> np.ndarray:
        original_h, original_w = img.shape[:2]
        target_h, target_w = self.image_size
        target_aspect = target_w / target_h
        img_aspect = original_w / original_h

        # Determine the scaling factor while respecting aspect ratio
        if img_aspect > target_aspect:  # Image is wider than target
            scale = target_w / original_w
        else:  # Image is taller than target
            scale = target_h / original_h

        # Resize the image with the scaling factor
        resized_w = int(original_w * scale)
        resized_h = int(original_h * scale)
        resized_img = cv2.resize(img, (resized_w, resized_h))

        # Center the resized image on the target canvas size with black padding
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        start_x = (target_w - resized_w) // 2
        start_y = (target_h - resized_h) // 2
        canvas[start_y:start_y + resized_h, start_x:start_x + resized_w] = resized_img

        if self.use_mediapipe:
            # Return the canvas as is for MediaPipe (uint8 format)
            return canvas
        else:
            # Normalize the image to [0, 1] for training when MediaPipe is not used
            return canvas.astype(np.float32) / 255.0

    def _print_details(self) -> None:
        num_batches = len(self)
        sequence_count = len(self.sequence_paths)
        sequence_length = self.sequence_length
        print(f"{num_batches} batches created, each of size {self.batch_size}, with {sequence_count} sequences of {sequence_length} images each.", f"Use MediaPipe: {self.use_mediapipe}")

    def on_epoch_end(self) -> None:
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(self.sequence_paths)

def visualize_sequence_with_keypoints(train_generator, images_per_row=5):
    """
    Visualizes a sequence of frames with MediaPipe keypoints overlaid, displayed in a grid.

    Args:
    - train_generator: A generator providing batches of data.
    - images_per_row: Number of images to display per row.
    """
    # Randomly select a batch index
    random_index = random.randint(0, len(train_generator) - 1)

    # Get the batch
    X, y = train_generator[random_index]

    # Extract the original frames for the first sequence in the batch
    random_sequence_name = train_generator.sequence_paths[random_index * train_generator.batch_size]
    random_sequence_path = os.path.join(train_generator.data_path, random_sequence_name)
    random_frames = sorted([
        os.path.join(random_sequence_path, img) for img in os.listdir(random_sequence_path)
        if img.endswith(('.jpg', '.png'))
    ])
    random_frames = [
        cv2.resize(cv2.imread(frame), train_generator.image_size) 
        for frame in random_frames[:train_generator.sequence_length]
    ]

    # Visualize the sequence
    num_frames = len(random_frames)
    num_rows = (num_frames + images_per_row - 1) // images_per_row  # Calculate rows needed

    plt.figure(figsize=(images_per_row * 4, num_rows * 4))  # Adjust figure size
    for i, (frame, keypoints) in enumerate(zip(random_frames, X[0])):
        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Overlay keypoints
        for x, y, _ in keypoints:
            # Scale and cast coordinates
            x_coord = int(x * frame.shape[1])  # Scale x to image width
            y_coord = int(y * frame.shape[0])  # Scale y to image height
            cv2.circle(frame_rgb, (x_coord, y_coord), 5, (0, 255, 0), -1)
        
        # Add subplot for the current frame
        plt.subplot(num_rows, images_per_row, i + 1)
        plt.imshow(frame_rgb)
        plt.title(f"Frame {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def get_callbacks(
    filepath: str,
    metric: str = 'accuracy',
    patience_lr: int = 5,
    patience_es: int = 10,
    min_lr: float = 1e-6,
    save_best_only=True
) -> Tuple[ModelCheckpoint, ReduceLROnPlateau, EarlyStopping]:
    """
    Creates and returns a set of common callbacks for training deep learning models.

    Args:
        filepath (str): Path to save the best model.
        metric (str): Metric to monitor (default is 'accuracy').
        patience_lr (int): Patience for learning rate reduction.
        patience_es (int): Patience for early stopping.
        min_lr (float): Minimum learning rate for ReduceLROnPlateau.

    Returns:
        Tuple[ModelCheckpoint, ReduceLROnPlateau, EarlyStopping]:
            - ModelCheckpoint: Saves the best model based on validation metric.
            - ReduceLROnPlateau: Reduces the learning rate when validation loss plateaus.
            - EarlyStopping: Stops training early if validation metric does not improve.
    """
    # Checkpoint to save the best model
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        monitor=f'val_{metric}',
        save_best_only=save_best_only,
        mode='max',
        verbose=0
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1
    )

    # Early stopping based on validation accuracy
    early_stopping_callback = EarlyStopping(
        monitor=f'val_{metric}',
        patience=patience_es,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )

    return checkpoint_callback, reduce_lr_callback, early_stopping_callback

def visualize_mediapipe_processing(generator):
    """
    Visualizes MediaPipe processing by overlaying keypoints on the original images for a random sequence.

    Args:
        generator (GestureDataGenerator): The data generator to sample from.

    Returns:
        None. Displays a plot of the processed frames.
    """
    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    # Select a random sequence
    random_index = random.randint(0, len(generator) - 1)
    X_batch, y_batch = generator[random_index]
    random_sequence_index = random.randint(0, len(X_batch) - 1)
    original_images = generator.sequence_paths[random_index * generator.batch_size + random_sequence_index]

    # Get the sequence directory
    sequence_dir = os.path.join(generator.data_path, original_images)

    # List and sort frame filenames alphabetically
    frame_files = sorted(
        [f for f in os.listdir(sequence_dir) if f.endswith(('.jpg', '.png'))]
    )

    # Extract the sequence data and keypoints
    keypoints_sequence = X_batch[random_sequence_index]
    label = y_batch[random_sequence_index]

    # Visualize the sequence
    num_frames = generator.sequence_length
    _, axes = plt.subplots(nrows=(num_frames // 5), ncols=5, figsize=(20, (num_frames // 5) * 4))

    for i, ax in enumerate(axes.flat[:num_frames]):
        if i >= len(frame_files):
            break  # Avoid out-of-range errors if fewer frames are available

        # Construct the path to the frame
        frame_path = os.path.join(sequence_dir, frame_files[i])

        # Load the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Frame not found: {frame_path}")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Overlay keypoints if available
        if keypoints_sequence is not None:
            keypoints = keypoints_sequence[i]
            for x, y, z in keypoints:  # (x, y, z) coordinates
                x_pixel = int(x * frame.shape[1])
                y_pixel = int(y * frame.shape[0])
                cv2.circle(frame, (x_pixel, y_pixel), radius=2, color=(0, 255, 0), thickness=-1)

        ax.imshow(frame)
        ax.axis("off")

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
