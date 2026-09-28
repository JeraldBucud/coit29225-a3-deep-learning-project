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
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Keep the large UTKFace dataset outside the Git repository.
# Path.home() makes the program portable between Windows user accounts
# while still following the required Desktop/UTKFace directory structure.
DATASET_PATH = Path.home() / "Desktop" / "UTKFace"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42


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
    image_files = sorted(dataset_path.glob("*.jpg"))

    for image_path in image_files:
        record = parse_filename(image_path)

        if record is None:
            invalid_files.append(image_path.name)
        else:
            records.append(record)

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

    ages = [record["age"] for record in records]
    genders = [record["gender"] for record in records]
    ethnicities = [record["ethnicity"] for record in records]

    print("\nUTKFace Dataset Summary")
    print("-----------------------")
    print(f"Total image files: {len(records) + len(invalid_files)}")
    print(f"Valid records: {len(records)}")
    print(f"Invalid records: {len(invalid_files)}")

    # Age statistics provide an initial view of the regression target.
    print("\nAge statistics:")
    print(f"Minimum age: {min(ages)}")
    print(f"Maximum age: {max(ages)}")
    print(f"Average age: {sum(ages) / len(ages):.2f}")

    # Gender counts help identify whether the classification target
    # contains a substantial class imbalance.
    print("\nGender distribution:")
    print(f"Gender 0: {genders.count(0)}")
    print(f"Gender 1: {genders.count(1)}")

    # Ethnicity is not an output target for this assessment, but retaining
    # its metadata is useful for validating the UTKFace filename structure.
    print("\nEthnicity distribution:")
    for ethnicity in range(5):
        print(f"Ethnicity {ethnicity}: {ethnicities.count(ethnicity)}")

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

    A fixed random seed is used so that the same split is produced between
    program runs. This makes later comparisons between CNN configurations
    and data augmentation experiments more consistent.

    Parameters
    ----------
    records : list
        Valid UTKFace metadata records.

    Returns
    -------
    tuple
        Training records and validation records.
    """

    train_records, validation_records = train_test_split(
        records,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    return train_records, validation_records


# ---------------------------------------------------------------------------
# 6. Main Program
# ---------------------------------------------------------------------------

def main():
    """Run the current UTKFace preprocessing and exploration stages."""

    if not DATASET_PATH.exists():
        print(f"Dataset folder not found: {DATASET_PATH}")
        return

    records, invalid_files = load_metadata(DATASET_PATH)

    display_dataset_summary(records, invalid_files)

    train_records, validation_records = split_records(records)

    print("\nDataset Split")
    print("-------------")
    print(f"Training records: {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")


if __name__ == "__main__":
    main()