"""
COIT29225 Neural Networks & Deep Learning
Assessment 3 - Task 2.1
CNN Age and Gender Estimation Using the UTKFace Dataset

This program develops a Convolutional Neural Network (CNN) for two
related prediction tasks using facial images from the UTKFace dataset:

1. Gender classification
2. Age estimation using regression

The complete program will perform the following stages:

- Load and validate UTKFace image metadata.
- Extract age, gender, and ethnicity information from filenames.
- Exclude malformed or invalid dataset records.
- Split the dataset into training and validation subsets.
- Resize and normalise facial images.
- Visually inspect preprocessed facial images.
- Build a shared CNN feature-extraction network.
- Create separate output heads for gender classification and age regression.
- Train the CNN using the original dataset.
- Train the CNN using data augmentation.
- Compare different architectural and algorithmic parameter configurations.
- Evaluate age prediction using Mean Absolute Error (MAE).
- Evaluate gender prediction using classification accuracy.
- Generate training and validation curves.
- Display sample predictions containing true and predicted age/gender labels.
- Generate a visual representation of the CNN architecture.

The UTKFace dataset is stored outside the Git repository and is expected
to be located in the user's Desktop/UTKFace directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Keep the large UTKFace dataset outside the Git repository.
# Path.home() makes the program portable between Windows user accounts
# while still following the required Desktop/UTKFace directory structure.
DATASET_PATH = Path.home() / "Desktop" / "UTKFace"

# Directory used for generated figures and other program outputs.
OUTPUT_DIR = Path("outputs")

# All images are resized to a consistent shape before entering the CNN.
IMAGE_SIZE = (128, 128)

# Number of images processed together during training and validation.
BATCH_SIZE = 32

# Reserve 20% of the validated dataset for model validation.
VALIDATION_SPLIT = 0.20

# A fixed random seed keeps dataset splitting and shuffling reproducible.
RANDOM_SEED = 42

# Data augmentation settings used for the augmented CNN experiment.
AUGMENTATION_ROTATION = 0.05
AUGMENTATION_ZOOM = 0.10
AUGMENTATION_CONTRAST = 0.10

# Initial optimisation settings for the baseline CNN experiment.
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.30
DENSE_UNITS = 128

# Small development-only training run used to verify that the complete
# model and data pipeline work before long experiments are started.
SMOKE_TEST_TRAIN_BATCHES = 5
SMOKE_TEST_VALIDATION_BATCHES = 2

# Number of epochs used for preliminary CNN parameter experiments.
EXPERIMENT_EPOCHS = 5

# Number of epochs used later for the selected final configuration.
FINAL_TRAINING_EPOCHS = 10


# ---------------------------------------------------------------------------
# 2. UTKFace Metadata Parsing
# ---------------------------------------------------------------------------

def parse_filename(image_path):
    """
    Extract and validate metadata encoded in a UTKFace image filename.

    A standard UTKFace filename begins with:

        age_gender_ethnicity_timestamp

    Parameters
    ----------
    image_path : pathlib.Path
        Path to a UTKFace image.

    Returns
    -------
    dict or None
        Dictionary containing the image path, age, gender, and ethnicity
        when the filename is valid. Returns None when the filename is
        malformed or contains values outside the expected ranges.
    """

    # Metadata fields are separated using underscores in UTKFace filenames.
    parts = image_path.name.split("_")

    # A valid record requires at least age, gender, ethnicity, and
    # the remaining timestamp/file information.
    if len(parts) < 4:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        ethnicity = int(parts[2])

    # Some dataset files may contain incomplete or malformed metadata.
    # These records are excluded instead of guessing missing values.
    except ValueError:
        return None

    # UTKFace ages are expected to be within the documented 0-116 range.
    if not 0 <= age <= 116:
        return None

    # Gender is represented as a binary value.
    if gender not in (0, 1):
        return None

    # UTKFace contains five ethnicity categories represented by 0-4.
    if ethnicity not in range(5):
        return None

    return {
        "image_path": image_path,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
    }


# ---------------------------------------------------------------------------
# 3. Dataset Metadata Loading
# ---------------------------------------------------------------------------

def load_metadata(dataset_path):
    """
    Load and validate metadata for every UTKFace image.

    Valid records are stored separately from malformed filenames so that
    invalid data can be reported without interrupting program execution.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Directory containing the UTKFace image files.

    Returns
    -------
    tuple
        A tuple containing:
        - list of valid metadata records
        - list of invalid filenames
    """

    records = []
    invalid_files = []

    # Sorting provides a consistent processing order between program runs.
    image_files = sorted(
        dataset_path.glob("*.jpg")
    )

    for image_path in image_files:
        record = parse_filename(
            image_path
        )

        if record is None:
            invalid_files.append(
                image_path.name
            )
        else:
            records.append(
                record
            )

    return records, invalid_files


# ---------------------------------------------------------------------------
# 4. Dataset Exploration
# ---------------------------------------------------------------------------

def display_dataset_summary(records, invalid_files):
    """
    Display descriptive statistics for the validated UTKFace dataset.

    The summary provides an initial understanding of the age, gender,
    and ethnicity distributions before image preprocessing and CNN training.

    Parameters
    ----------
    records : list
        Valid UTKFace metadata records.

    invalid_files : list
        Filenames that could not be parsed or validated.
    """

    ages = [
        record["age"]
        for record in records
    ]

    genders = [
        record["gender"]
        for record in records
    ]

    ethnicities = [
        record["ethnicity"]
        for record in records
    ]

    print("\nUTKFace Dataset Summary")
    print("-----------------------")
    print(
        f"Total image files: "
        f"{len(records) + len(invalid_files)}"
    )
    print(
        f"Valid records: "
        f"{len(records)}"
    )
    print(
        f"Invalid records: "
        f"{len(invalid_files)}"
    )

    # Age statistics provide an initial view of the regression target.
    print("\nAge statistics:")
    print(
        f"Minimum age: "
        f"{min(ages)}"
    )
    print(
        f"Maximum age: "
        f"{max(ages)}"
    )
    print(
        f"Average age: "
        f"{sum(ages) / len(ages):.2f}"
    )

    # Gender counts help identify whether the classification target
    # contains a substantial class imbalance.
    print("\nGender distribution:")
    print(
        f"Gender 0: "
        f"{genders.count(0)}"
    )
    print(
        f"Gender 1: "
        f"{genders.count(1)}"
    )

    # Ethnicity is not an output target for this assessment, but retaining
    # its metadata is useful for validating the UTKFace filename structure.
    print("\nEthnicity distribution:")

    for ethnicity in range(5):
        print(
            f"Ethnicity {ethnicity}: "
            f"{ethnicities.count(ethnicity)}"
        )

    # Report malformed records explicitly so preprocessing decisions
    # remain transparent and reproducible.
    if invalid_files:
        print("\nInvalid filenames:")

        for filename in invalid_files:
            print(filename)


# ---------------------------------------------------------------------------
# 5. Dataset Splitting
# ---------------------------------------------------------------------------

def split_records(records):
    """
    Split validated UTKFace records into training and validation subsets.

    Gender labels are used for stratification so that both subsets retain
    approximately the same gender distribution. A fixed random seed keeps
    the split reproducible between experiments.

    Parameters
    ----------
    records : list
        Valid UTKFace metadata records.

    Returns
    -------
    tuple
        Training records and validation records.
    """

    gender_labels = [
        record["gender"]
        for record in records
    ]

    train_records, validation_records = train_test_split(
        records,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=gender_labels,
    )

    return train_records, validation_records


# ---------------------------------------------------------------------------
# 6. Image Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path, age, gender):
    """
    Load and preprocess one UTKFace image for CNN input.

    Each image is decoded as RGB, resized to the configured image size,
    and normalised from the original 0-255 pixel range to 0-1.

    Parameters
    ----------
    image_path : tf.Tensor
        Path to the image file.

    age : tf.Tensor
        Ground-truth age value.

    gender : tf.Tensor
        Ground-truth gender label.

    Returns
    -------
    tuple
        Preprocessed image tensor and a dictionary containing the
        corresponding age and gender targets.
    """

    # Read the JPEG file from disk only when TensorFlow requires it.
    image = tf.io.read_file(
        image_path
    )

    # Decode the image using three colour channels (RGB).
    image = tf.io.decode_jpeg(
        image,
        channels=3,
    )

    # Resize every image to a consistent shape required by the CNN.
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    # Convert pixel values from 0-255 to floating-point values from 0-1.
    image = tf.cast(
        image,
        tf.float32,
    ) / 255.0

    # Store both prediction targets using names that will later match
    # the two output heads of the multi-output CNN.
    labels = {
        "gender": tf.cast(
            gender,
            tf.float32,
        ),
        "age": tf.cast(
            age,
            tf.float32,
        ),
    }

    return image, labels


def create_tf_dataset(records, training=False):
    """
    Create a TensorFlow data pipeline from validated UTKFace records.

    Images are loaded lazily from disk rather than storing the complete
    image dataset in memory. Training data is shuffled while validation
    data remains in a stable order.

    Parameters
    ----------
    records : list
        Valid UTKFace metadata records.

    training : bool, optional
        Whether the dataset is being prepared for training.

    Returns
    -------
    tf.data.Dataset
        Batched and prefetched TensorFlow dataset.
    """

    image_paths = [
        str(record["image_path"])
        for record in records
    ]

    ages = [
        record["age"]
        for record in records
    ]

    genders = [
        record["gender"]
        for record in records
    ]

    # Construct a TensorFlow dataset from paths and target labels.
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            ages,
            genders,
        )
    )

    # Shuffle only the training dataset. A fixed seed keeps experiments
    # reproducible while reshuffling the order between training epochs.
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    # Decode, resize, and normalise images in parallel.
    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Group individual images into training batches.
    dataset = dataset.batch(
        BATCH_SIZE
    )

    # Prepare future batches while the current batch is being processed.
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ---------------------------------------------------------------------------
# 7. Preprocessed Image Visualisation
# ---------------------------------------------------------------------------

def visualise_preprocessed_images(dataset, num_images=9):
    """
    Display and save a sample of preprocessed UTKFace images.

    This visual check confirms that images remain visually meaningful after
    JPEG decoding, resizing, RGB conversion, and pixel normalisation.

    Parameters
    ----------
    dataset : tf.data.Dataset
        TensorFlow dataset containing preprocessed images and labels.

    num_images : int, optional
        Number of images to display. The default is 9.
    """

    # Retrieve one batch from the dataset for visual inspection.
    images, labels = next(
        iter(dataset)
    )

    # Ensure the requested sample does not exceed the batch size.
    num_images = min(
        num_images,
        int(images.shape[0]),
    )

    # Use three columns to provide a compact visual arrangement.
    columns = 3

    rows = (
        num_images
        + columns
        - 1
    ) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8, 8),
    )
    # Flatten the subplot array so each position can be accessed using
    # a single index.
    axes = axes.flatten()

    for index in range(num_images):
        image = images[index].numpy()

        age = int(
            labels["age"][index].numpy()
        )

        gender = int(
            labels["gender"][index].numpy()
        )

        axes[index].imshow(
            image
        )

        axes[index].set_title(
            f"Age: {age} | Gender: {gender}"
        )

        axes[index].axis(
            "off"
        )

    # Hide unused subplot positions if the requested sample does not
    # completely fill the final row.
    for index in range(
        num_images,
        len(axes),
    ):
        axes[index].axis(
            "off"
        )

    figure.suptitle(
        "Preprocessed UTKFace Samples",
        fontsize=14,
    )

    figure.tight_layout(
    rect=(0, 0, 1, 0.95),
    h_pad=1.2,
    w_pad=0.8,
    )

    # Create the output directory only when generated files are required.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "utkface_preprocessed_samples.png"
    )

    # Save a high-resolution copy that can later be used as report evidence.
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("\nPreprocessed Image Visualisation")
    print("--------------------------------")
    print(
        f"Saved figure: "
        f"{output_path}"
    )

    # Display the figure interactively during development.
    plt.show()

    # Release the Matplotlib figure after it has been displayed.
    plt.close(
        figure
    )



# ---------------------------------------------------------------------------
# 8. Data Augmentation
# ---------------------------------------------------------------------------

def create_data_augmentation():
    """
    Create the image augmentation pipeline used during CNN training.

    The transformations are intentionally conservative so that facial
    characteristics important for age and gender prediction are preserved.

    Augmentation is applied only during training and is disabled
    automatically during validation and prediction.

    Returns
    -------
    tf.keras.Sequential
        Keras preprocessing pipeline containing the augmentation layers.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal"
            ),
            tf.keras.layers.RandomRotation(
                AUGMENTATION_ROTATION
            ),
            tf.keras.layers.RandomZoom(
                AUGMENTATION_ZOOM
            ),
            tf.keras.layers.RandomContrast(
                AUGMENTATION_CONTRAST
            ),

            # Keep augmented values within the same 0-1 range used by
            # the normal preprocessing pipeline.
            tf.keras.layers.ReLU(
                max_value=1.0,
                name="clip_augmented_pixels",
            ),
        ],
        name="data_augmentation",
    )

def visualise_augmented_images(dataset, augmentation, num_images=9):
    """
    Display and save augmented versions of preprocessed UTKFace images.

    This visual inspection helps confirm that augmentation introduces useful
    variation without excessively distorting facial characteristics.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Preprocessed TensorFlow dataset.

    augmentation : tf.keras.Sequential
        Data augmentation pipeline.

    num_images : int, optional
        Number of augmented images to display. The default is 9.
    """

    images, labels = next(
        iter(dataset)
    )

    num_images = min(
        num_images,
        int(images.shape[0]),
    )

    # Force augmentation during this visual test.
    augmented_images = augmentation(
        images[:num_images],
        training=True,
    )

    columns = 3

    rows = (
        num_images
        + columns
        - 1
    ) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8, 8),
    )

    axes = axes.flatten()

    for index in range(num_images):
        age = int(
            labels["age"][index].numpy()
        )

        gender = int(
            labels["gender"][index].numpy()
        )

        axes[index].imshow(
            augmented_images[index].numpy()
        )

        axes[index].set_title(
            f"Age: {age} | Gender: {gender}"
        )

        axes[index].axis(
            "off"
        )

    for index in range(
        num_images,
        len(axes),
    ):
        axes[index].axis(
            "off"
        )

    figure.suptitle(
        "Augmented UTKFace Samples",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.95),
        h_pad=1.2,
        w_pad=0.8,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "utkface_augmented_samples.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("\nData Augmentation Visualisation")
    print("-------------------------------")
    print(
        f"Saved figure: "
        f"{output_path}"
    )

    plt.show()

    plt.close(
        figure
    )



# ---------------------------------------------------------------------------
# 9. CNN Architecture
# ---------------------------------------------------------------------------

def build_cnn_model(
    use_augmentation=False,
    filters=(32, 64, 128),
    dense_units=128,
    dropout_rate=0.30,
):
    """
    Build the multi-output CNN for gender classification and age regression.

    The model uses a shared convolutional feature-extraction backbone.
    After the Flatten layer, the network separates into two task-specific
    branches:

    - Gender head: binary classification
    - Age head: regression

    Parameters
    ----------
    use_augmentation : bool, optional
        Whether to include the data augmentation pipeline at the beginning
        of the model. The default is False.

    filters : tuple, optional
        Number of convolutional filters used in each convolution block.
        The default configuration is (32, 64, 128).

    dense_units : int, optional
        Number of neurons in each task-specific fully connected layer.
        The default is 128.

    dropout_rate : float, optional
        Fraction of neurons randomly disabled during training to help
        reduce overfitting. The default is 0.30.

    Returns
    -------
    tf.keras.Model
        Multi-output CNN with gender and age prediction heads.
    """

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        name="image",
    )

    x = inputs

    # Data augmentation is included only for the augmented experiment.
    # Keras automatically disables these random transformations during
    # validation and prediction.
    if use_augmentation:
        augmentation = create_data_augmentation()
        x = augmentation(x)

    # Shared convolutional feature-extraction backbone.
    for block_number, number_of_filters in enumerate(
        filters,
        start=1,
    ):
        x = tf.keras.layers.Conv2D(
            filters=number_of_filters,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name=f"conv_{block_number}",
        )(x)

        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            name=f"pool_{block_number}",
        )(x)

    # Convert the final convolutional feature maps into a one-dimensional
    # feature vector before splitting into the two prediction heads.
    shared_features = tf.keras.layers.Flatten(
        name="flatten",
    )(x)

    # ------------------------------------------------------------------
    # Gender classification head
    # ------------------------------------------------------------------

    gender_branch = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="gender_dense",
    )(shared_features)

    gender_branch = tf.keras.layers.Dropout(
        dropout_rate,
        name="gender_dropout",
    )(gender_branch)

    # Sigmoid produces a probability suitable for binary classification.
    gender_output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="gender",
    )(gender_branch)

    # ------------------------------------------------------------------
    # Age regression head
    # ------------------------------------------------------------------

    age_branch = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="age_dense",
    )(shared_features)

    age_branch = tf.keras.layers.Dropout(
        dropout_rate,
        name="age_dropout",
    )(age_branch)

    # A linear output allows the model to predict a continuous age value.
    age_output = tf.keras.layers.Dense(
        1,
        activation="linear",
        name="age",
    )(age_branch)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "gender": gender_output,
            "age": age_output,
        },
        name=(
            "utkface_cnn_augmented"
            if use_augmentation
            else "utkface_cnn"
        ),
    )

    return model



# ---------------------------------------------------------------------------
# 10. Model Compilation
# ---------------------------------------------------------------------------

def compile_cnn_model(model, learning_rate=LEARNING_RATE):
    """
    Compile the multi-output CNN with task-specific losses and metrics.

    Gender prediction is treated as binary classification and therefore
    uses binary cross-entropy loss with classification accuracy.

    Age prediction is treated as regression and therefore uses mean
    squared error as the optimisation loss and mean absolute error as
    an interpretable evaluation metric.

    Parameters
    ----------
    model : tf.keras.Model
        CNN containing the gender classification and age regression heads.

    learning_rate : float, optional
        Learning rate used by the Adam optimiser.

    Returns
    -------
    tf.keras.Model
        Compiled CNN ready for training.
    """

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "gender": "binary_crossentropy",
            "age": "mse",
        },
        metrics={
            "gender": [
                tf.keras.metrics.BinaryAccuracy(
                    name="accuracy"
                )
            ],
            "age": [
                tf.keras.metrics.MeanAbsoluteError(
                    name="mae"
                )
            ],
        },
    )

    return model


# ---------------------------------------------------------------------------
# 11. Training Smoke Test
# ---------------------------------------------------------------------------

def run_training_smoke_test(
    model,
    train_dataset,
    validation_dataset,
):
    """
    Run a short training test to verify the complete CNN pipeline.

    Only a few batches are used so that model compilation, label mapping,
    forward propagation, backpropagation, losses, and metrics can be checked
    before longer experimental training runs are started.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled multi-output CNN.

    train_dataset : tf.data.Dataset
        Preprocessed training dataset.

    validation_dataset : tf.data.Dataset
        Preprocessed validation dataset.

    Returns
    -------
    tf.keras.callbacks.History
        Keras training history produced by the smoke test.
    """

    print("\nCNN Training Smoke Test")
    print("-----------------------")
    print(
        f"Training batches: "
        f"{SMOKE_TEST_TRAIN_BATCHES}"
    )
    print(
        f"Validation batches: "
        f"{SMOKE_TEST_VALIDATION_BATCHES}"
    )

    history = model.fit(
        train_dataset.take(
            SMOKE_TEST_TRAIN_BATCHES
        ),
        validation_data=validation_dataset.take(
            SMOKE_TEST_VALIDATION_BATCHES
        ),
        epochs=1,
        shuffle=False,
        verbose=1,
    )

    return history



# ---------------------------------------------------------------------------
# 12. CNN Experiment Configuration
# ---------------------------------------------------------------------------

CNN_CONFIGURATIONS = [
    {
        "name": "A_baseline",
        "filters": (32, 64, 128),
        "dense_units": 128,
        "dropout_rate": 0.30,
        "learning_rate": 0.001,
    },
    {
        "name": "B_compact",
        "filters": (16, 32, 64),
        "dense_units": 64,
        "dropout_rate": 0.30,
        "learning_rate": 0.001,
    },
    {
        "name": "C_wider",
        "filters": (32, 64, 128, 256),
        "dense_units": 128,
        "dropout_rate": 0.30,
        "learning_rate": 0.001,
    },
    {
        "name": "D_regularised",
        "filters": (32, 64, 128),
        "dense_units": 128,
        "dropout_rate": 0.50,
        "learning_rate": 0.0005,
    },
]

def display_cnn_configurations():
    """
    Display the CNN configurations prepared for parameter experiments.

    The configurations vary architectural and algorithmic parameters so
    their effects on age regression and gender classification performance
    can be compared systematically.
    """

    print("\nCNN Experiment Configurations")
    print("-----------------------------")

    for configuration in CNN_CONFIGURATIONS:
        print(
            f"\n{configuration['name']}"
        )
        print(
            f"Filters: "
            f"{configuration['filters']}"
        )
        print(
            f"Dense units: "
            f"{configuration['dense_units']}"
        )
        print(
            f"Dropout rate: "
            f"{configuration['dropout_rate']}"
        )
        print(
            f"Learning rate: "
            f"{configuration['learning_rate']}"
        )

    print(
        f"\nPreliminary experiment epochs: "
        f"{EXPERIMENT_EPOCHS}"
    )

# ---------------------------------------------------------------------------
# 13. Main Program
# ---------------------------------------------------------------------------

def main():
    """Run the current UTKFace preprocessing and exploration stages."""

    # Confirm that the external UTKFace dataset is available.
    if not DATASET_PATH.exists():
        print(
            f"Dataset folder not found: "
            f"{DATASET_PATH}"
        )
        return

    # Load and validate metadata encoded in the image filenames.
    records, invalid_files = load_metadata(
        DATASET_PATH
    )

    # Display descriptive information about the validated dataset.
    display_dataset_summary(
        records,
        invalid_files,
    )

    # Split the valid records into reproducible training and
    # validation subsets.
    train_records, validation_records = split_records(
        records
    )

    print("\nDataset Split")
    print("-------------")
    print(
        f"Training records: "
        f"{len(train_records)}"
    )
    print(
        f"Validation records: "
        f"{len(validation_records)}"
    )

    # Build TensorFlow input pipelines for both subsets.
    train_dataset = create_tf_dataset(
        train_records,
        training=True,
    )

    validation_dataset = create_tf_dataset(
        validation_records,
        training=False,
    )

    print("\nTensorFlow Dataset Test")
    print("-----------------------")

    # Inspect one training batch to confirm image preprocessing,
    # target preparation, batching, and normalisation.
    train_images, train_labels = next(
        iter(train_dataset)
    )

    print("\nTraining batch:")
    print(
        f"Image batch shape: "
        f"{train_images.shape}"
    )
    print(
        f"Gender batch shape: "
        f"{train_labels['gender'].shape}"
    )
    print(
        f"Age batch shape: "
        f"{train_labels['age'].shape}"
    )
    print(
        f"Minimum pixel value: "
        f"{tf.reduce_min(train_images).numpy():.4f}"
    )
    print(
        f"Maximum pixel value: "
        f"{tf.reduce_max(train_images).numpy():.4f}"
    )

    # Inspect one validation batch to verify that the validation
    # pipeline produces the same input and target structure.
    validation_images, validation_labels = next(
        iter(validation_dataset)
    )

    print("\nValidation batch:")
    print(
        f"Image batch shape: "
        f"{validation_images.shape}"
    )
    print(
        f"Gender batch shape: "
        f"{validation_labels['gender'].shape}"
    )
    print(
        f"Age batch shape: "
        f"{validation_labels['age'].shape}"
    )
    print(
        f"Minimum pixel value: "
        f"{tf.reduce_min(validation_images).numpy():.4f}"
    )
    print(
        f"Maximum pixel value: "
        f"{tf.reduce_max(validation_images).numpy():.4f}"
    )

    # Display a stable validation sample after preprocessing. Validation
    # data is used because it is not shuffled between program runs.
    visualise_preprocessed_images(
        validation_dataset,
        num_images=9,
    )

        # Build the augmentation pipeline for the augmented CNN experiment.
    augmentation = create_data_augmentation()

    # Visually inspect augmented samples before using them for training.
    visualise_augmented_images(
        validation_dataset,
        augmentation,
        num_images=9,
    )

    # Build the baseline CNN without data augmentation.
    model = build_cnn_model(
        use_augmentation=False,
        filters=(32, 64, 128),
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
    )

    # Configure task-specific losses, metrics, and optimisation.
    model = compile_cnn_model(
        model,
        learning_rate=LEARNING_RATE,
    )

    print("\nCNN Architecture")
    print("----------------")

    model.summary()

    # Run a short development test before starting full CNN experiments.
    run_training_smoke_test(
        model,
        train_dataset,
        validation_dataset,
    )

    # Display the parameter configurations that will be evaluated
    # during the CNN tuning stage.
    display_cnn_configurations()

if __name__ == "__main__":
    main()