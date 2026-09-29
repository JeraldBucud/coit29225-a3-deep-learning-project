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
- Explicitly evaluate trained models on both training and validation sets.
- Generate age MAE, gender accuracy, age loss, and gender loss curves.
- Generate ten sample age/gender predictions for each executed model run.
- Save trained models and experiment histories for later evaluation.
- Provide the assessment-required three-option command-line menu.

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

# UTKFace gender encoding used throughout prediction output.
# 0 = Male, 1 = Female.
GENDER_LABELS = {
    0: "Male",
    1: "Female",
}


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
# 13. CNN Evaluation
# ---------------------------------------------------------------------------

def evaluate_cnn_model(
    model,
    train_records,
    validation_records,
    batch_size,
):
    """
    Explicitly evaluate a trained CNN on the training and validation sets.

    Evaluation datasets are created without shuffling. For augmented models,
    Keras augmentation layers are automatically disabled because evaluation
    runs with ``training=False``.

    Parameters
    ----------
    model : tf.keras.Model
        Trained and compiled multi-output CNN.

    train_records : list
        Metadata records assigned to the training subset.

    validation_records : list
        Metadata records assigned to the validation subset.

    batch_size : int
        Batch size used during evaluation.

    Returns
    -------
    tuple
        Training metric dictionary and validation metric dictionary.
    """

    training_evaluation_dataset = create_tf_dataset(
        train_records,
        training=False,
        batch_size=batch_size,
    )

    validation_evaluation_dataset = create_tf_dataset(
        validation_records,
        training=False,
        batch_size=batch_size,
    )

    training_metrics = model.evaluate(
        training_evaluation_dataset,
        verbose=0,
        return_dict=True,
    )

    validation_metrics = model.evaluate(
        validation_evaluation_dataset,
        verbose=0,
        return_dict=True,
    )

    print("\nCNN Evaluation")
    print("==============")

    print("\nTraining Set")
    print("------------")
    print(
        f"Age MAE: "
        f"{training_metrics['age_mae']:.4f} years"
    )
    print(
        f"Gender accuracy: "
        f"{training_metrics['gender_accuracy']:.4f} "
        f"({training_metrics['gender_accuracy'] * 100:.2f}%)"
    )
    print(
        f"Age loss: "
        f"{training_metrics['age_loss']:.4f}"
    )
    print(
        f"Gender loss: "
        f"{training_metrics['gender_loss']:.4f}"
    )

    print("\nValidation Set")
    print("--------------")
    print(
        f"Age MAE: "
        f"{validation_metrics['age_mae']:.4f} years"
    )
    print(
        f"Gender accuracy: "
        f"{validation_metrics['gender_accuracy']:.4f} "
        f"({validation_metrics['gender_accuracy'] * 100:.2f}%)"
    )
    print(
        f"Age loss: "
        f"{validation_metrics['age_loss']:.4f}"
    )
    print(
        f"Gender loss: "
        f"{validation_metrics['gender_loss']:.4f}"
    )

    return training_metrics, validation_metrics


# ---------------------------------------------------------------------------
# 14. CNN Sample Predictions
# ---------------------------------------------------------------------------

def generate_cnn_sample_predictions(
    model,
    validation_records,
    output_prefix,
    num_samples=10,
    show_plot=True,
):
    """
    Generate reproducible age and gender predictions for validation images.

    The function accepts any trained CNN model. This allows every parameter
    configuration and every final program run to produce the ten sample
    predictions required by the assessment specification.

    Parameters
    ----------
    model : tf.keras.Model
        Trained CNN used for inference.

    validation_records : list
        Metadata records assigned to the validation subset.

    output_prefix : str
        Prefix used for the generated PNG and CSV filenames.

    num_samples : int, optional
        Number of validation images to predict.

    show_plot : bool, optional
        Whether to display the prediction figure interactively.

    Returns
    -------
    pandas.DataFrame
        Prediction results for the selected validation images.
    """

    num_samples = min(
        num_samples,
        len(validation_records),
    )

    random_generator = Random(
        RANDOM_SEED
    )

    sample_records = random_generator.sample(
        validation_records,
        num_samples,
    )

    sample_dataset = create_tf_dataset(
        sample_records,
        training=False,
        batch_size=num_samples,
    )

    images, labels = next(
        iter(sample_dataset)
    )

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
        actual_gender = int(
            actual_genders[index]
        )

        predicted_gender = int(
            predicted_genders[index]
        )

        prediction_rows.append(
            {
                "filename": (
                    sample_records[index][
                        "image_path"
                    ].name
                ),
                "actual_age": int(
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
                "actual_gender": actual_gender,
                "actual_gender_label": (
                    GENDER_LABELS[
                        actual_gender
                    ]
                ),
                "predicted_gender": (
                    predicted_gender
                ),
                "predicted_gender_label": (
                    GENDER_LABELS[
                        predicted_gender
                    ]
                ),
                "gender_probability": round(
                    float(
                        gender_probabilities[
                            index
                        ]
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
        / f"{output_prefix}_sample_predictions.csv"
    )

    results_table.to_csv(
        csv_path,
        index=False,
    )

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

    # plt.subplots returns an ndarray for the normal ten-image case.
    # Converting to a flat array keeps indexing straightforward.
    axes = axes.flatten()

    for index in range(num_samples):
        actual_gender = int(
            actual_genders[index]
        )

        predicted_gender = int(
            predicted_genders[index]
        )

        axes[index].imshow(
            images[index].numpy()
        )

        title = (
            f"Age: {actual_ages[index]} "
            f"→ {predicted_ages[index]:.1f}\n"
            f"{GENDER_LABELS[actual_gender]} "
            f"→ "
            f"{GENDER_LABELS[predicted_gender]}"
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
        / f"{output_prefix}_sample_predictions.png"
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
                "actual_gender_label",
                "predicted_gender_label",
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

    if show_plot:
        plt.show()

    plt.close(
        figure
    )

    return results_table


# ---------------------------------------------------------------------------
# 15. CNN Experiment Runner
# ---------------------------------------------------------------------------

def run_cnn_experiments(
    train_records,
    validation_records,
    configurations,
):
    """
    Train and evaluate multiple CNN parameter configurations.

    Every configuration uses the same validated training/validation split.
    After training, the model is explicitly evaluated and ten reproducible
    sample predictions are generated, as required by the specification.

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

        training_metrics, validation_metrics = evaluate_cnn_model(
            model,
            train_records,
            validation_records,
            configuration[
                "batch_size"
            ],
        )

        output_prefix = (
            f"cnn_experiment_"
            f"{configuration['name']}"
        )

        generate_cnn_sample_predictions(
            model,
            validation_records,
            output_prefix=output_prefix,
            num_samples=10,
            show_plot=False,
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
            "evaluated_train_age_mae": round(
                training_metrics[
                    "age_mae"
                ],
                4,
            ),
            "evaluated_train_gender_accuracy": round(
                training_metrics[
                    "gender_accuracy"
                ],
                4,
            ),
            "evaluated_validation_age_mae": round(
                validation_metrics[
                    "age_mae"
                ],
                4,
            ),
            "evaluated_validation_gender_accuracy": round(
                validation_metrics[
                    "gender_accuracy"
                ],
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
# 16. Single-Run Training Curves
# ---------------------------------------------------------------------------

def generate_single_run_training_plots(
    history,
    scenario,
    show_plot=True,
):
    """
    Generate the four required training/validation curves for one CNN run.

    Parameters
    ----------
    history : dict
        Keras training-history dictionary.

    scenario : str
        Run identifier such as ``original`` or ``augmented``.

    show_plot : bool, optional
        Whether to display the combined curve figure.

    Returns
    -------
    pathlib.Path
        Path to the saved combined training-curve figure.
    """

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    epochs = range(
        1,
        len(history["age_mae"]) + 1,
    )

    # Age MAE.
    axes[0, 0].plot(
        epochs,
        history["age_mae"],
        label="Training",
    )

    axes[0, 0].plot(
        epochs,
        history["val_age_mae"],
        label="Validation",
    )

    axes[0, 0].set_title(
        "Age Mean Absolute Error"
    )

    axes[0, 0].set_xlabel(
        "Epoch"
    )

    axes[0, 0].set_ylabel(
        "MAE (Years)"
    )

    axes[0, 0].legend()
    axes[0, 0].grid(
        alpha=0.3
    )

    # Gender accuracy.
    axes[0, 1].plot(
        epochs,
        history["gender_accuracy"],
        label="Training",
    )

    axes[0, 1].plot(
        epochs,
        history["val_gender_accuracy"],
        label="Validation",
    )

    axes[0, 1].set_title(
        "Gender Classification Accuracy"
    )

    axes[0, 1].set_xlabel(
        "Epoch"
    )

    axes[0, 1].set_ylabel(
        "Accuracy"
    )

    axes[0, 1].legend()
    axes[0, 1].grid(
        alpha=0.3
    )

    # Age loss.
    axes[1, 0].plot(
        epochs,
        history["age_loss"],
        label="Training",
    )

    axes[1, 0].plot(
        epochs,
        history["val_age_loss"],
        label="Validation",
    )

    axes[1, 0].set_title(
        "Age Regression Loss"
    )

    axes[1, 0].set_xlabel(
        "Epoch"
    )

    axes[1, 0].set_ylabel(
        "Mean Squared Error"
    )

    axes[1, 0].legend()
    axes[1, 0].grid(
        alpha=0.3
    )

    # Gender loss.
    axes[1, 1].plot(
        epochs,
        history["gender_loss"],
        label="Training",
    )

    axes[1, 1].plot(
        epochs,
        history["val_gender_loss"],
        label="Validation",
    )

    axes[1, 1].set_title(
        "Gender Classification Loss"
    )

    axes[1, 1].set_xlabel(
        "Epoch"
    )

    axes[1, 1].set_ylabel(
        "Binary Cross-Entropy"
    )

    axes[1, 1].legend()
    axes[1, 1].grid(
        alpha=0.3
    )

    figure.suptitle(
        f"CNN Training Results - "
        f"{scenario.capitalize()} Dataset",
        fontsize=15,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"cnn_{scenario}_training_curves.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"\nTraining curves saved to: "
        f"{output_path}"
    )

    if show_plot:
        plt.show()

    plt.close(
        figure
    )

    return output_path


# ---------------------------------------------------------------------------
# 17. Final Comparison Curves
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
    Save a training/validation curve comparing original and augmented CNNs.
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    Generate original-vs-augmented comparison curves for the final report.
    """

    print("\nGenerating Final CNN Comparison Curves")
    print("--------------------------------------")

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
# 18. CNN Architecture Visualisation
# ---------------------------------------------------------------------------

def generate_cnn_architecture_visualisation(
    model,
    output_prefix,
    show_plot=True,
):
    """
    Generate and save a graph representation of a CNN architecture.

    Parameters
    ----------
    model : tf.keras.Model
        CNN whose architecture should be visualised.

    output_prefix : str
        Prefix used for the output filename.

    show_plot : bool, optional
        Whether the architecture image should be displayed.

    Returns
    -------
    pathlib.Path
        Path to the generated architecture image.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{output_prefix}_architecture_graph.png"
    )

    architecture_image = visualkeras.graph_view(
        model,
        to_file=str(output_path),
    )

    print("\nCNN Architecture Visualisation")
    print("------------------------------")
    print(
        f"Architecture saved to: "
        f"{output_path}"
    )

    if show_plot:
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
# 19. Final CNN Training
# ---------------------------------------------------------------------------

def train_final_cnn(
    train_records,
    validation_records,
    configuration,
    use_augmentation=False,
):
    """
    Train, evaluate, and report the selected final CNN configuration.

    The original-data and augmented runs use the same architecture,
    optimiser settings, split, and epoch count. The controlled difference
    is whether data augmentation is enabled.

    Returns
    -------
    tuple
        Training-history dictionary and summary-result dictionary.
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

    print("\nCNN Architecture")
    print("----------------")
    model.summary()

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

    training_metrics, validation_metrics = evaluate_cnn_model(
        model,
        train_records,
        validation_records,
        configuration["batch_size"],
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
        "evaluated_train_age_mae": round(
            training_metrics[
                "age_mae"
            ],
            4,
        ),
        "evaluated_train_gender_accuracy": round(
            training_metrics[
                "gender_accuracy"
            ],
            4,
        ),
        "evaluated_validation_age_mae": round(
            validation_metrics[
                "age_mae"
            ],
            4,
        ),
        "evaluated_validation_gender_accuracy": round(
            validation_metrics[
                "gender_accuracy"
            ],
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

    output_prefix = (
        f"cnn_{scenario}"
    )

    generate_single_run_training_plots(
        history.history,
        scenario,
        show_plot=True,
    )

    generate_cnn_architecture_visualisation(
        model,
        output_prefix=output_prefix,
        show_plot=True,
    )

    generate_cnn_sample_predictions(
        model,
        validation_records,
        output_prefix=output_prefix,
        num_samples=10,
        show_plot=True,
    )

    return history.history, result


# ---------------------------------------------------------------------------
# 20. Final Original-vs-Augmented Comparison
# ---------------------------------------------------------------------------

def run_final_cnn_comparison(
    train_records,
    validation_records,
):
    """
    Train and compare the selected CNN with and without augmentation.

    This helper is retained for reproducible report generation. The normal
    assessment interface runs the two scenarios individually through the
    three-option menu.

    Returns
    -------
    pandas.DataFrame
        Final comparison results for both scenarios.
    """

    histories = {}
    results = []

    original_history, original_result = train_final_cnn(
        train_records,
        validation_records,
        SELECTED_CNN_CONFIGURATION,
        use_augmentation=False,
    )

    histories["original"] = (
        original_history
    )

    results.append(
        original_result
    )

    augmented_history, augmented_result = train_final_cnn(
        train_records,
        validation_records,
        SELECTED_CNN_CONFIGURATION,
        use_augmentation=True,
    )

    histories["augmented"] = (
        augmented_history
    )

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
# 21. Application Menu
# ---------------------------------------------------------------------------

def display_main_menu():
    """
    Display the three-option menu required by the assessment specification.
    """

    print(
        "\nCOIT29225 Assessment 3"
    )

    print(
        "CNN Age and Gender Estimation"
    )

    print(
        "============================="
    )

    print(
        "1. Run CNN"
    )

    print(
        "2. Run CNN with data augmentation"
    )

    print(
        "3. Quit"
    )


def run_menu(
    train_records,
    validation_records,
):
    """
    Run the assessment-required CNN application menu.

    Option 1 trains and evaluates the selected CNN using the original
    preprocessed dataset.

    Option 2 trains and evaluates the same CNN configuration with data
    augmentation.

    Option 3 exits the application.
    """

    while True:
        display_main_menu()

        choice = input(
            "\nSelect an option (1-3): "
        ).strip()

        if choice == "1":
            print(
                "\nRunning CNN using "
                "the original dataset..."
            )

            train_final_cnn(
                train_records,
                validation_records,
                SELECTED_CNN_CONFIGURATION,
                use_augmentation=False,
            )

        elif choice == "2":
            print(
                "\nRunning CNN with "
                "data augmentation..."
            )

            train_final_cnn(
                train_records,
                validation_records,
                SELECTED_CNN_CONFIGURATION,
                use_augmentation=True,
            )

        elif choice == "3":
            print(
                "\nExiting application."
            )

            break

        else:
            print(
                "\nInvalid selection. "
                "Please enter 1, 2, or 3."
            )


# ---------------------------------------------------------------------------
# 22. Main Program
# ---------------------------------------------------------------------------

def main():
    """
    Load UTKFace metadata and launch the required CNN application menu.
    """

    if not DATASET_PATH.exists():
        print(
            f"\nDataset folder not found: "
            f"{DATASET_PATH}"
        )

        return

    print(
        "\nLoading UTKFace dataset..."
    )

    records, invalid_files = load_metadata(
        DATASET_PATH
    )

    if not records:
        print(
            "No valid UTKFace images were found."
        )

        return

    train_records, validation_records = split_records(
        records
    )

    print(
        f"Valid images: "
        f"{len(records):,}"
    )

    print(
        f"Invalid images: "
        f"{len(invalid_files):,}"
    )

    print(
        f"Training images: "
        f"{len(train_records):,}"
    )

    print(
        f"Validation images: "
        f"{len(validation_records):,}"
    )

    run_menu(
        train_records,
        validation_records,
    )


if __name__ == "__main__":
    main()
