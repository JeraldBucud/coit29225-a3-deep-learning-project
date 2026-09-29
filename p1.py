"""
COIT29225 Neural Networks & Deep Learning
Assessment 3 - Task 2.1
CNN Age and Gender Estimation Using the UTKFace Dataset

This program develops a Convolutional Neural Network (CNN) for two
related prediction tasks using facial images from the UTKFace dataset:

1. Gender classification
2. Age estimation using regression

The program includes the following stages:

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
- Run a short smoke test to validate the training pipeline.
- Compare architectural and algorithmic parameter configurations.
- Record parameter experiment results.
- Select a CNN configuration based on the tuning results.
- Train the selected CNN using the original dataset.
- Train the same CNN using data augmentation.
- Compare original and augmented CNN performance.
- Generate age MAE, gender accuracy, age loss, and gender loss curves.
- Save trained models and experiment histories for later evaluation.

The UTKFace dataset is stored outside the Git repository and is expected
to be located in the user's Desktop/UTKFace directory.
"""

from pathlib import Path
from random import Random
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import visualkeras
from sklearn.model_selection import train_test_split



# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Keep the large UTKFace dataset outside the Git repository.
DATASET_PATH = Path.home() / "Desktop" / "UTKFace"

# Directory used for generated figures, models, and experiment outputs.
OUTPUT_DIR = Path("outputs")

# All images are resized to a consistent shape before entering the CNN.
IMAGE_SIZE = (128, 128)

# Default number of images processed together in each batch.
BATCH_SIZE = 32

# Reserve 20% of the validated dataset for validation.
VALIDATION_SPLIT = 0.20

# Fixed seed used to improve reproducibility between experiments.
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

# Number of epochs used during preliminary parameter experiments.
EXPERIMENT_EPOCHS = 5

# Number of epochs used for the final original-vs-augmentation comparison.
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

    # A valid record requires age, gender, ethnicity, and timestamp data.
    if len(parts) < 4:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        ethnicity = int(parts[2])

    except ValueError:
        return None

    # UTKFace ages are expected to be within the documented range.
    if not 0 <= age <= 116:
        return None

    # Gender is represented as a binary label.
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
    invalid data can be reported without interrupting execution.

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

    # Sorting provides a consistent processing order between runs.
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

    # Ethnicity is not a prediction target for this assessment,
    # but it is retained for validating UTKFace metadata.
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

    Images are loaded lazily from disk rather than storing the entire
    image dataset in memory.

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

    # Training data is shuffled in the input pipeline.
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

    A shared convolutional backbone extracts facial features. After the
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

    # Augmentation is only included in the augmented experiment.
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

    # The assessment specifies that the network splits after Flatten.
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
        Binary cross-entropy loss and classification accuracy.

    Age:
        Mean squared error loss and mean absolute error metric.

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
    Run a short development training test for the complete CNN pipeline.

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


# Configuration A provided the strongest overall balance between
# age-regression and gender-classification performance during tuning.
SELECTED_CNN_CONFIGURATION = CNN_CONFIGURATIONS[0]


def display_cnn_configurations():
    """Display CNN configurations prepared for parameter experiments."""

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

        tf.keras.backend.clear_session()

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
            "configuration": configuration["name"],
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
# 14. Final CNN Training
# ---------------------------------------------------------------------------

def train_final_cnn(
    train_records,
    validation_records,
    configuration,
    use_augmentation=False,
):
    """
    Train the selected CNN configuration for the final comparison.

    The same architecture, optimiser settings, dataset split, and epoch
    count are used in both scenarios. The controlled difference is whether
    data augmentation is enabled.

    Parameters
    ----------
    train_records : list
        Training metadata records.

    validation_records : list
        Validation metadata records.

    configuration : dict
        Selected CNN parameter configuration.

    use_augmentation : bool, optional
        Whether augmentation should be included during training.

    Returns
    -------
    tuple
        Training history dictionary and summary result dictionary.
    """

    scenario = (
        "augmented"
        if use_augmentation
        else "original"
    )

    print("\n" + "=" * 60)

    print(
        f"Final CNN Training: "
        f"{scenario.capitalize()} Dataset"
    )

    print("=" * 60)

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    train_dataset = create_tf_dataset(
        train_records,
        training=True,
        batch_size=configuration["batch_size"],
    )

    validation_dataset = create_tf_dataset(
        validation_records,
        training=False,
        batch_size=configuration["batch_size"],
    )

    model = build_cnn_model(
        use_augmentation=use_augmentation,
        filters=configuration["filters"],
        dense_units=configuration["dense_units"],
        dropout_rate=configuration["dropout_rate"],
    )

    model = compile_cnn_model(
        model,
        learning_rate=configuration["learning_rate"],
    )

    print(
        f"Configuration: "
        f"{configuration['name']}"
    )

    print(
        f"Data augmentation: "
        f"{use_augmentation}"
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
        f"Epochs: "
        f"{FINAL_TRAINING_EPOCHS}"
    )

    print(
        f"Parameters: "
        f"{model.count_params():,}"
    )

    start_time = perf_counter()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=FINAL_TRAINING_EPOCHS,
        shuffle=False,
        verbose=1,
    )

    training_time = (
        perf_counter()
        - start_time
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        OUTPUT_DIR
        / f"cnn_{scenario}.keras"
    )

    model.save(
        model_path
    )

    history_table = pd.DataFrame(
        history.history
    )

    history_table.insert(
        0,
        "epoch",
        range(
            1,
            len(history_table) + 1,
        ),
    )

    history_path = (
        OUTPUT_DIR
        / f"cnn_{scenario}_history.csv"
    )

    history_table.to_csv(
        history_path,
        index=False,
    )

    best_age_epoch = (
        history.history["val_age_mae"].index(
            min(
                history.history[
                    "val_age_mae"
                ]
            )
        )
        + 1
    )

    best_gender_epoch = (
        history.history[
            "val_gender_accuracy"
        ].index(
            max(
                history.history[
                    "val_gender_accuracy"
                ]
            )
        )
        + 1
    )

    result = {
        "scenario": scenario,
        "configuration": configuration["name"],
        "augmentation": use_augmentation,
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
        "epochs": FINAL_TRAINING_EPOCHS,
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
        "best_age_epoch": best_age_epoch,
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
        "best_gender_epoch": best_gender_epoch,
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

    print(
        f"\nCompleted final "
        f"{scenario} training."
    )

    print(
        f"Best validation age MAE: "
        f"{result['best_val_age_mae']:.4f} "
        f"(epoch {best_age_epoch})"
    )

    print(
        f"Best validation gender accuracy: "
        f"{result['best_val_gender_accuracy']:.4f} "
        f"(epoch {best_gender_epoch})"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Model saved to: "
        f"{model_path}"
    )

    print(
        f"History saved to: "
        f"{history_path}"
    )

    return history.history, result


# ---------------------------------------------------------------------------
# 15. Final Training Curves
# ---------------------------------------------------------------------------

def save_metric_comparison_plot(
    histories,
    metric,
    validation_metric,
    title,
    y_label,
    filename,
):
    """
    Save a training and validation curve comparing final CNN scenarios.

    Parameters
    ----------
    histories : dict
        Histories for the original and augmented scenarios.

    metric : str
        Training metric key.

    validation_metric : str
        Validation metric key.

    title : str
        Plot title.

    y_label : str
        Vertical-axis label.

    filename : str
        Output PNG filename.
    """

    figure = plt.figure(
        figsize=(8, 6)
    )

    for scenario, history in histories.items():
        epochs = range(
            1,
            len(history[metric]) + 1,
        )

        label_name = scenario.capitalize()

        plt.plot(
            epochs,
            history[metric],
            label=f"{label_name} Training",
        )

        plt.plot(
            epochs,
            history[validation_metric],
            linestyle="--",
            label=f"{label_name} Validation",
        )

    plt.title(
        title
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        y_label
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / filename
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved figure: "
        f"{output_path}"
    )


def generate_final_training_plots(histories):
    """
    Generate the CNN curves required for the assessment report.

    The following comparisons are generated:

    - Age Mean Absolute Error
    - Gender classification accuracy
    - Age regression loss
    - Gender classification loss
    """

    print("\nGenerating Final CNN Curves")
    print("---------------------------")

    save_metric_comparison_plot(
        histories,
        metric="age_mae",
        validation_metric="val_age_mae",
        title="CNN Age Mean Absolute Error",
        y_label="Mean Absolute Error (Years)",
        filename="cnn_age_mae_comparison.png",
    )

    save_metric_comparison_plot(
        histories,
        metric="gender_accuracy",
        validation_metric="val_gender_accuracy",
        title="CNN Gender Classification Accuracy",
        y_label="Accuracy",
        filename="cnn_gender_accuracy_comparison.png",
    )

    save_metric_comparison_plot(
        histories,
        metric="age_loss",
        validation_metric="val_age_loss",
        title="CNN Age Regression Loss",
        y_label="Mean Squared Error",
        filename="cnn_age_loss_comparison.png",
    )

    save_metric_comparison_plot(
        histories,
        metric="gender_loss",
        validation_metric="val_gender_loss",
        title="CNN Gender Classification Loss",
        y_label="Binary Cross-Entropy",
        filename="cnn_gender_loss_comparison.png",
    )


# ---------------------------------------------------------------------------
# 16. Final Original-vs-Augmented Comparison
# ---------------------------------------------------------------------------

def run_final_cnn_comparison(
    train_records,
    validation_records,
):
    """
    Train and compare the selected CNN with and without augmentation.

    Returns
    -------
    pandas.DataFrame
        Final comparison results for both scenarios.
    """

    histories = {}
    results = []

    # Scenario 1: original preprocessed dataset.
    original_history, original_result = train_final_cnn(
        train_records,
        validation_records,
        SELECTED_CNN_CONFIGURATION,
        use_augmentation=False,
    )

    histories["original"] = original_history

    results.append(
        original_result
    )

    # Scenario 2: same CNN with training-time augmentation.
    augmented_history, augmented_result = train_final_cnn(
        train_records,
        validation_records,
        SELECTED_CNN_CONFIGURATION,
        use_augmentation=True,
    )

    histories["augmented"] = augmented_history

    results.append(
        augmented_result
    )

    results_table = pd.DataFrame(
        results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        OUTPUT_DIR
        / "cnn_final_comparison.csv"
    )

    results_table.to_csv(
        comparison_path,
        index=False,
    )

    print("\nFinal CNN Comparison")
    print("====================")

    print(
        results_table.to_string(
            index=False
        )
    )

    print(
        f"\nComparison saved to: "
        f"{comparison_path}"
    )

    generate_final_training_plots(
        histories
    )

    return results_table

# ---------------------------------------------------------------------------
# 17. CNN Architecture Visualisation
# ---------------------------------------------------------------------------

def generate_cnn_architecture_visualisation():
    """
    Generate and save a visual representation of the trained CNN.

    The graph-based Visualkeras renderer is used because the CNN contains
    a shared convolutional backbone followed by separate age-regression
    and gender-classification branches.

    The saved original-data model is used so architecture generation does
    not require the network to be retrained.

    Returns
    -------
    pathlib.Path or None
        Path to the generated architecture image, or None if the saved
        model is unavailable.
    """

    model_path = (
        OUTPUT_DIR
        / "cnn_original.keras"
    )

    if not model_path.exists():
        print("\nCNN Architecture Visualisation")
        print("------------------------------")
        print(
            f"Saved model not found: "
            f"{model_path}"
        )

        return None

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "cnn_architecture_graph.png"
    )

    # Load the trained model without restoring optimisation state because
    # only its network structure is required for visualisation.
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    # Graph view preserves the branching structure of the multi-output CNN.
    architecture_image = visualkeras.graph_view(
        model,
        to_file=str(output_path),
    )

    print("\nCNN Architecture Visualisation")
    print("------------------------------")
    print(
        f"Loaded model: "
        f"{model_path}"
    )
    print(
        f"Architecture saved to: "
        f"{output_path}"
    )

    # Display the generated architecture at a manageable window size.
    figure = plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        architecture_image
    )

    plt.title(
        "UTKFace Multi-Output CNN Architecture"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.show()

    plt.close(
        figure
    )

    return output_path


# ---------------------------------------------------------------------------
# 18. CNN Sample Predictions
# ---------------------------------------------------------------------------

def generate_cnn_sample_predictions(
    validation_records,
    num_samples=10,
):
    """
    Generate age and gender predictions for validation images.

    A reproducible random sample is selected from the validation subset.
    The saved original-data CNN is loaded so predictions can be generated
    without retraining the network.

    For each sample, the figure shows:

    - Actual age
    - Predicted age
    - Actual gender
    - Predicted gender

    A CSV file containing the numerical prediction results is also saved.

    Parameters
    ----------
    validation_records : list
        Metadata records assigned to the validation subset.

    num_samples : int, optional
        Number of validation images to predict. The default is 10.

    Returns
    -------
    pandas.DataFrame or None
        Prediction results, or None when the saved model is unavailable.
    """

    model_path = (
        OUTPUT_DIR
        / "cnn_original.keras"
    )

    if not model_path.exists():
        print("\nCNN Sample Predictions")
        print("----------------------")
        print(
            f"Saved model not found: "
            f"{model_path}"
        )

        return None

    num_samples = min(
        num_samples,
        len(validation_records),
    )

    # Select a reproducible random validation sample.
    random_generator = Random(
        RANDOM_SEED
    )

    sample_records = random_generator.sample(
        validation_records,
        num_samples,
    )

    # Create a dataset without shuffling so image order remains aligned
    # with the selected metadata records.
    sample_dataset = create_tf_dataset(
        sample_records,
        training=False,
        batch_size=num_samples,
    )

    images, labels = next(
        iter(sample_dataset)
    )

    # Load the trained original-data model without restoring its optimiser.
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    # Perform inference only. Augmentation and dropout are disabled.
    predictions = model(
        images,
        training=False,
    )

    predicted_ages = (
        predictions["age"]
        .numpy()
        .reshape(-1)
    )

    gender_probabilities = (
        predictions["gender"]
        .numpy()
        .reshape(-1)
    )

    predicted_genders = (
        gender_probabilities >= 0.5
    ).astype(int)

    actual_ages = (
        labels["age"]
        .numpy()
        .astype(int)
    )

    actual_genders = (
        labels["gender"]
        .numpy()
        .astype(int)
    )

    prediction_rows = []

    for index in range(num_samples):
        prediction_rows.append(
            {
                "filename": (
                    sample_records[index][
                        "image_path"
                    ].name
                ),
                "actual_age": (
                    actual_ages[index]
                ),
                "predicted_age": round(
                    float(
                        predicted_ages[index]
                    ),
                    2,
                ),
                "absolute_age_error": round(
                    abs(
                        float(
                            predicted_ages[index]
                        )
                        - actual_ages[index]
                    ),
                    2,
                ),
                "actual_gender": (
                    actual_genders[index]
                ),
                "predicted_gender": (
                    predicted_genders[index]
                ),
                "gender_probability": round(
                    float(
                        gender_probabilities[index]
                    ),
                    4,
                ),
            }
        )

    results_table = pd.DataFrame(
        prediction_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR
        / "cnn_sample_predictions.csv"
    )

    results_table.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Prediction visualisation
    # ---------------------------------------------------------------

    columns = 5

    rows = (
        num_samples
        + columns
        - 1
    ) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 7),
    )

    axes = axes.flatten()

    for index in range(num_samples):
        axes[index].imshow(
            images[index].numpy()
        )

        title = (
            f"Age: {actual_ages[index]} "
            f"→ {predicted_ages[index]:.1f}\n"
            f"Gender: {actual_genders[index]} "
            f"→ {predicted_genders[index]}"
        )

        axes[index].set_title(
            title,
            fontsize=10,
        )

        axes[index].axis(
            "off"
        )

    for index in range(
        num_samples,
        len(axes),
    ):
        axes[index].axis(
            "off"
        )

    figure.suptitle(
        "CNN Age and Gender Predictions",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.95),
        h_pad=1.5,
        w_pad=1.0,
    )

    figure_path = (
        OUTPUT_DIR
        / "cnn_sample_predictions.png"
    )

    figure.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("\nCNN Sample Predictions")
    print("----------------------")

    print(
        results_table[
            [
                "actual_age",
                "predicted_age",
                "absolute_age_error",
                "actual_gender",
                "predicted_gender",
                "gender_probability",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nPrediction figure saved to: "
        f"{figure_path}"
    )

    print(
        f"Prediction data saved to: "
        f"{csv_path}"
    )

    plt.show()

    plt.close(
        figure
    )

    return results_table


# ---------------------------------------------------------------------------
# 19. Main Program
# ---------------------------------------------------------------------------

def main():
    """Run the current UTKFace CNN development and evaluation pipeline."""

    if not DATASET_PATH.exists():
        print(
            f"Dataset folder not found: "
            f"{DATASET_PATH}"
        )

        return

    # ------------------------------------------------------------------
    # Dataset loading and exploration
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Verify TensorFlow input pipelines
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Development visualisations
    # ------------------------------------------------------------------
    #
    # These functions remain available but are not displayed automatically
    # during long training runs.
    #
    # Uncomment when visual inspection is required.
    #
    # visualise_preprocessed_images(
    #     validation_dataset,
    #     num_images=9,
    # )
    #
    # augmentation = create_data_augmentation()
    #
    # visualise_augmented_images(
    #     validation_dataset,
    #     augmentation,
    #     num_images=9,
    # )

    # ------------------------------------------------------------------
    # Display selected baseline architecture
    # ------------------------------------------------------------------

    model = build_cnn_model(
        use_augmentation=False,
        filters=SELECTED_CNN_CONFIGURATION[
            "filters"
        ],
        dense_units=SELECTED_CNN_CONFIGURATION[
            "dense_units"
        ],
        dropout_rate=SELECTED_CNN_CONFIGURATION[
            "dropout_rate"
        ],
    )

    model = compile_cnn_model(
        model,
        learning_rate=SELECTED_CNN_CONFIGURATION[
            "learning_rate"
        ],
    )

    print("\nCNN Architecture")
    print("----------------")

    model.summary()

    # ------------------------------------------------------------------
    # Development smoke test
    # ------------------------------------------------------------------
    #
    # The smoke test has already verified that the model can perform
    # forward propagation, backpropagation, and validation successfully.
    #
    # run_training_smoke_test(
    #     model,
    #     train_dataset,
    #     validation_dataset,
    # )

    # ------------------------------------------------------------------
    # Parameter configurations
    # ------------------------------------------------------------------

    display_cnn_configurations()

    # Parameter tuning has already been completed.
    # Uncomment this only when parameter experiments need to be repeated.
    #
    # run_cnn_experiments(
    #     train_records,
    #     validation_records,
    #     CNN_CONFIGURATIONS,
    # )

    # Final original-vs-augmentation training has already been completed.
    # Uncomment only if the final models need to be retrained.
    #
    # run_final_cnn_comparison(
    #     train_records,
    #     validation_records,
    # )
    
    # Generate the CNN architecture diagram from the saved final model.
    generate_cnn_architecture_visualisation()


    # Generate ten reproducible prediction examples from the
    # validation subset using the saved final original-data model.
    generate_cnn_sample_predictions(
        validation_records,
        num_samples=10,
    )

    


    

if __name__ == "__main__":
    main()