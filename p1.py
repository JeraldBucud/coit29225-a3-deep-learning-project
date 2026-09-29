"""
COIT29225 Neural Networks & Deep Learning
Assessment 3 - Task 2.1
CNN Age and Gender Estimation Using the UTKFace Dataset

This program develops a Convolutional Neural Network (CNN) for two
related prediction tasks using facial images from the UTKFace dataset:

1. Gender classification
2. Age estimation using regression

The complete program performs the following stages:

- Load and validate UTKFace image metadata.
- Extract age, gender, and ethnicity information from filenames.
- Exclude malformed or invalid dataset records.
- Split the dataset into training and validation subsets.
- Resize and normalise facial images.
- Visually inspect preprocessed facial images.
- Define a controlled image data augmentation pipeline.
- Build a shared CNN feature-extraction network.
- Create separate output heads for gender classification and age regression.
- Compile the model using task-specific losses and evaluation metrics.
- Run a short smoke test to validate the complete training pipeline.
- Compare multiple architectural and algorithmic parameter configurations.
- Record experiment results for later analysis.
- Train the CNN using the original dataset.
- Train the CNN using data augmentation.
- Evaluate age prediction using Mean Absolute Error (MAE).
- Evaluate gender prediction using classification accuracy.
- Generate training and validation curves.
- Display sample predictions containing true and predicted age/gender labels.
- Generate a visual representation of the CNN architecture.

The UTKFace dataset is stored outside the Git repository and is expected
to be located in the user's Desktop/UTKFace directory.
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Keep the large UTKFace dataset outside the Git repository.
# Path.home() makes the program portable between Windows user accounts
# while still following the required Desktop/UTKFace directory structure.
DATASET_PATH = Path.home() / "Desktop" / "UTKFace"

# Directory used for generated figures and experiment outputs.
OUTPUT_DIR = Path("outputs")

# All images are resized to a consistent shape before entering the CNN.
IMAGE_SIZE = (128, 128)

# Default number of images processed together during training/validation.
BATCH_SIZE = 32

# Reserve 20% of the validated dataset for model validation.
VALIDATION_SPLIT = 0.20

# A fixed random seed keeps splitting and experiments reproducible.
RANDOM_SEED = 42

# Conservative augmentation settings for facial images.
AUGMENTATION_ROTATION = 0.05
AUGMENTATION_ZOOM = 0.10
AUGMENTATION_CONTRAST = 0.10

# Baseline CNN optimisation settings.
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.30
DENSE_UNITS = 128

# Development-only smoke-test settings.
SMOKE_TEST_TRAIN_BATCHES = 5
SMOKE_TEST_VALIDATION_BATCHES = 2

# Preliminary training duration used when comparing CNN configurations.
EXPERIMENT_EPOCHS = 5

# Longer training duration intended for the selected final configuration.
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
        Dictionary containing image path, age, gender, and ethnicity when
        valid. Returns None when the filename is malformed or contains
        values outside the expected ranges.
    """

    parts = image_path.name.split("_")

    # Valid filenames require age, gender, ethnicity, and timestamp data.
    if len(parts) < 4:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        ethnicity = int(parts[2])
    except ValueError:
        return None

    # UTKFace documents ages from 0 to 116.
    if not 0 <= age <= 116:
        return None

    # Gender is represented as a binary label.
    if gender not in (0, 1):
        return None

    # UTKFace contains five ethnicity codes: 0-4.
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

    Parameters
    ----------
    dataset_path : pathlib.Path
        Directory containing UTKFace image files.

    Returns
    -------
    tuple
        Valid metadata records and malformed/invalid filenames.
    """

    records = []
    invalid_files = []

    # Sorting provides a stable processing order between runs.
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

    print("\nGender distribution:")
    print(
        f"Gender 0: "
        f"{genders.count(0)}"
    )
    print(
        f"Gender 1: "
        f"{genders.count(1)}"
    )

    # Ethnicity is retained for dataset validation even though this
    # assessment does not use ethnicity as a prediction target.
    print("\nEthnicity distribution:")

    for ethnicity in range(5):
        print(
            f"Ethnicity {ethnicity}: "
            f"{ethnicities.count(ethnicity)}"
        )

    if invalid_files:
        print("\nInvalid filenames:")

        for filename in invalid_files:
            print(filename)


# ---------------------------------------------------------------------------
# 5. Dataset Splitting
# ---------------------------------------------------------------------------

def split_records(records):
    """
    Split validated records into training and validation subsets.

    Gender labels are used for stratification so both subsets retain
    approximately the same gender distribution.

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
    Load and preprocess one UTKFace image.

    Images are decoded as RGB, resized to IMAGE_SIZE, and normalised from
    the original 0-255 pixel range to 0-1.

    Parameters
    ----------
    image_path : tf.Tensor
        Path to the image file.

    age : tf.Tensor
        Ground-truth age.

    gender : tf.Tensor
        Ground-truth gender label.

    Returns
    -------
    tuple
        Preprocessed image and task-specific target dictionary.
    """

    image = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_jpeg(
        image,
        channels=3,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    ) / 255.0

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


def create_tf_dataset(
    records,
    training=False,
    batch_size=BATCH_SIZE,
):
    """
    Create a TensorFlow input pipeline from validated UTKFace records.

    Parameters
    ----------
    records : list
        Valid UTKFace metadata records.

    training : bool, optional
        Whether the dataset is used for training.

    batch_size : int, optional
        Number of images included in each batch.

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

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            ages,
            genders,
        )
    )

    # Training data is shuffled here rather than by model.fit().
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ---------------------------------------------------------------------------
# 7. Preprocessed Image Visualisation
# ---------------------------------------------------------------------------

def visualise_preprocessed_images(
    dataset,
    num_images=9,
):
    """
    Display and save preprocessed UTKFace image samples.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Preprocessed dataset.

    num_images : int, optional
        Number of images to display.
    """

    images, labels = next(
        iter(dataset)
    )

    num_images = min(
        num_images,
        int(images.shape[0]),
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "utkface_preprocessed_samples.png"
    )

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

    plt.show()

    plt.close(
        figure
    )


# ---------------------------------------------------------------------------
# 8. Data Augmentation
# ---------------------------------------------------------------------------

def create_data_augmentation():
    """
    Create the augmentation pipeline for the augmented CNN experiment.

    Conservative transformations are used so facial characteristics
    important for age and gender prediction remain meaningful.

    Returns
    -------
    tf.keras.Sequential
        Keras augmentation pipeline.
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

            # Keep augmented values in the same 0-1 range as normal images.
            tf.keras.layers.ReLU(
                max_value=1.0,
                name="clip_augmented_pixels",
            ),
        ],
        name="data_augmentation",
    )


def visualise_augmented_images(
    dataset,
    augmentation,
    num_images=9,
):
    """
    Display and save augmented versions of UTKFace images.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Preprocessed TensorFlow dataset.

    augmentation : tf.keras.Sequential
        Augmentation pipeline.

    num_images : int, optional
        Number of images to display.
    """

    images, labels = next(
        iter(dataset)
    )

    num_images = min(
        num_images,
        int(images.shape[0]),
    )

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
    Build the multi-output CNN.

    A shared convolutional backbone extracts image features. After the
    Flatten layer, the network separates into:

    - Gender binary-classification head
    - Age regression head

    Parameters
    ----------
    use_augmentation : bool, optional
        Whether augmentation should be included inside the model.

    filters : tuple, optional
        Number of convolutional filters in each block.

    dense_units : int, optional
        Number of neurons in each task-specific dense layer.

    dropout_rate : float, optional
        Dropout rate used in both task-specific branches.

    Returns
    -------
    tf.keras.Model
        Multi-output CNN.
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

    if use_augmentation:
        augmentation = create_data_augmentation()
        x = augmentation(
            x
        )

    # Shared convolution and pooling backbone.
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

    # The assessment requires the model to split after the Flatten layer.
    shared_features = tf.keras.layers.Flatten(
        name="flatten",
    )(x)

    # Gender classification branch.
    gender_branch = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="gender_dense",
    )(shared_features)

    gender_branch = tf.keras.layers.Dropout(
        dropout_rate,
        name="gender_dropout",
    )(gender_branch)

    gender_output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="gender",
    )(gender_branch)

    # Age regression branch.
    age_branch = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="age_dense",
    )(shared_features)

    age_branch = tf.keras.layers.Dropout(
        dropout_rate,
        name="age_dropout",
    )(age_branch)

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

def compile_cnn_model(
    model,
    learning_rate=LEARNING_RATE,
):
    """
    Compile the CNN using task-specific losses and metrics.

    Gender:
        Binary cross-entropy loss + classification accuracy.

    Age:
        Mean squared error loss + mean absolute error metric.

    Parameters
    ----------
    model : tf.keras.Model
        Multi-output CNN.

    learning_rate : float, optional
        Adam optimiser learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled CNN.
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
    Run a short training test for the complete CNN pipeline.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled CNN.

    train_dataset : tf.data.Dataset
        Training dataset.

    validation_dataset : tf.data.Dataset
        Validation dataset.

    Returns
    -------
    tf.keras.callbacks.History
        Keras training history.
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
        "batch_size": 32,
    },
    {
        "name": "B_compact",
        "filters": (16, 32, 64),
        "dense_units": 64,
        "dropout_rate": 0.30,
        "learning_rate": 0.001,
        "batch_size": 32,
    },
    {
        "name": "C_wider",
        "filters": (32, 64, 128, 256),
        "dense_units": 128,
        "dropout_rate": 0.30,
        "learning_rate": 0.001,
        "batch_size": 32,
    },
    {
        "name": "D_regularised",
        "filters": (32, 64, 128),
        "dense_units": 128,
        "dropout_rate": 0.50,
        "learning_rate": 0.0005,
        "batch_size": 64,
    },
]


def display_cnn_configurations():
    """Display the CNN parameter configurations prepared for experiments."""

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
            f"Batch size: "
            f"{configuration['batch_size']}"
        )

    print(
        f"\nPreliminary experiment epochs: "
        f"{EXPERIMENT_EPOCHS}"
    )


# ---------------------------------------------------------------------------
# 13. CNN Experiment Runner
# ---------------------------------------------------------------------------

def run_cnn_experiments(
    train_records,
    validation_records,
    configurations,
):
    """
    Train and evaluate multiple CNN parameter configurations.

    Each configuration receives a newly initialised CNN so that architectural
    and algorithmic settings can be compared fairly.

    Parameters
    ----------
    train_records : list
        Training metadata records.

    validation_records : list
        Validation metadata records.

    configurations : list
        CNN configurations to evaluate.

    Returns
    -------
    pandas.DataFrame
        Experiment parameter and evaluation results.
    """

    results = []

    print("\nCNN Parameter Experiments")
    print("=========================")

    for experiment_number, configuration in enumerate(
        configurations,
        start=1,
    ):
        print(
            f"\nExperiment {experiment_number}/"
            f"{len(configurations)}"
        )
        print(
            f"Configuration: "
            f"{configuration['name']}"
        )
        print("-" * 40)

        # Clear the previous TensorFlow/Keras model before starting
        # another configuration.
        tf.keras.backend.clear_session()

        # Reset randomness so experiments start under comparable conditions.
        tf.keras.utils.set_random_seed(
            RANDOM_SEED
        )

        experiment_train_dataset = create_tf_dataset(
            train_records,
            training=True,
            batch_size=configuration[
                "batch_size"
            ],
        )

        experiment_validation_dataset = create_tf_dataset(
            validation_records,
            training=False,
            batch_size=configuration[
                "batch_size"
            ],
        )

        model = build_cnn_model(
            use_augmentation=False,
            filters=configuration[
                "filters"
            ],
            dense_units=configuration[
                "dense_units"
            ],
            dropout_rate=configuration[
                "dropout_rate"
            ],
        )

        model = compile_cnn_model(
            model,
            learning_rate=configuration[
                "learning_rate"
            ],
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
            f"Dropout: "
            f"{configuration['dropout_rate']}"
        )
        print(
            f"Learning rate: "
            f"{configuration['learning_rate']}"
        )
        print(
            f"Batch size: "
            f"{configuration['batch_size']}"
        )
        print(
            f"Parameters: "
            f"{model.count_params():,}"
        )

        start_time = perf_counter()

        history = model.fit(
            experiment_train_dataset,
            validation_data=experiment_validation_dataset,
            epochs=EXPERIMENT_EPOCHS,
            shuffle=False,
            verbose=1,
        )

        training_time = (
            perf_counter()
            - start_time
        )

        result = {
            "configuration": configuration[
                "name"
            ],
            "filters": str(
                configuration["filters"]
            ),
            "dense_units": configuration[
                "dense_units"
            ],
            "dropout_rate": configuration[
                "dropout_rate"
            ],
            "learning_rate": configuration[
                "learning_rate"
            ],
            "batch_size": configuration[
                "batch_size"
            ],
            "epochs": EXPERIMENT_EPOCHS,
            "parameters": model.count_params(),
            "training_seconds": round(
                training_time,
                2,
            ),
            "final_val_age_mae": round(
                history.history[
                    "val_age_mae"
                ][-1],
                4,
            ),
            "best_val_age_mae": round(
                min(
                    history.history[
                        "val_age_mae"
                    ]
                ),
                4,
            ),
            "final_val_gender_accuracy": round(
                history.history[
                    "val_gender_accuracy"
                ][-1],
                4,
            ),
            "best_val_gender_accuracy": round(
                max(
                    history.history[
                        "val_gender_accuracy"
                    ]
                ),
                4,
            ),
            "final_val_age_loss": round(
                history.history[
                    "val_age_loss"
                ][-1],
                4,
            ),
            "final_val_gender_loss": round(
                history.history[
                    "val_gender_loss"
                ][-1],
                4,
            ),
            "final_val_total_loss": round(
                history.history[
                    "val_loss"
                ][-1],
                4,
            ),
        }

        results.append(
            result
        )

        print(
            f"\nCompleted "
            f"{configuration['name']}"
        )
        print(
            f"Best validation age MAE: "
            f"{result['best_val_age_mae']:.4f}"
        )
        print(
            f"Best validation gender accuracy: "
            f"{result['best_val_gender_accuracy']:.4f}"
        )
        print(
            f"Training time: "
            f"{training_time:.2f} seconds"
        )

    results_table = pd.DataFrame(
        results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "cnn_experiment_results.csv"
    )

    results_table.to_csv(
        output_path,
        index=False,
    )

    print("\nCNN Experiment Results")
    print("======================")

    print(
        results_table.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    return results_table


# ---------------------------------------------------------------------------
# 14. Main Program
# ---------------------------------------------------------------------------

def main():
    """Run the current UTKFace CNN development pipeline."""

    if not DATASET_PATH.exists():
        print(
            f"Dataset folder not found: "
            f"{DATASET_PATH}"
        )
        return

    records, invalid_files = load_metadata(
        DATASET_PATH
    )

    display_dataset_summary(
        records,
        invalid_files,
    )

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

    # Standard datasets are retained for pipeline validation and
    # development checks.
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

    # The preprocessing and augmentation visualisations have already been
    # validated. Their functions remain available but are not displayed
    # automatically during long parameter experiments.

    # visualise_preprocessed_images(
    #     validation_dataset,
    #     num_images=9,
    # )

    # augmentation = create_data_augmentation()

    # visualise_augmented_images(
    #     validation_dataset,
    #     augmentation,
    #     num_images=9,
    # )

    # Build and display the baseline architecture.
    model = build_cnn_model(
        use_augmentation=False,
        filters=(32, 64, 128),
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
    )

    model = compile_cnn_model(
        model,
        learning_rate=LEARNING_RATE,
    )

    print("\nCNN Architecture")
    print("----------------")

    model.summary()

    # The smoke test has already confirmed that the complete model and
    # label pipeline can perform forward and backward propagation.
    #
    # run_training_smoke_test(
    #     model,
    #     train_dataset,
    #     validation_dataset,
    # )

    display_cnn_configurations()

    # Development checkpoint:
    # Run only Configuration A before allowing every configuration to run.
    run_cnn_experiments(
        train_records,
        validation_records,
        CNN_CONFIGURATIONS,
    )


if __name__ == "__main__":
    main()