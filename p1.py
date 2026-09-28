from pathlib import Path


DATASET_PATH = Path.home() / "Desktop" / "UTKFace"


def parse_filename(image_path):
    """Extract UTKFace metadata from an image filename."""

    parts = image_path.name.split("_")

    # A valid UTKFace filename should contain:
    # age_gender_ethnicity_timestamp
    if len(parts) < 4:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        ethnicity = int(parts[2])
    except ValueError:
        return None

    # Validate metadata ranges.
    if not 0 <= age <= 116:
        return None

    if gender not in (0, 1):
        return None

    if ethnicity not in range(5):
        return None

    return {
        "image_path": image_path,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
    }


def load_metadata(dataset_path):
    """Load and validate metadata for all UTKFace images."""

    records = []
    invalid_files = []

    image_files = sorted(dataset_path.glob("*.jpg"))

    for image_path in image_files:
        record = parse_filename(image_path)

        if record is None:
            invalid_files.append(image_path.name)
        else:
            records.append(record)

    return records, invalid_files


def display_dataset_summary(records, invalid_files):
    """Display basic statistics about the UTKFace dataset."""

    ages = [record["age"] for record in records]
    genders = [record["gender"] for record in records]
    ethnicities = [record["ethnicity"] for record in records]

    print("\nUTKFace Dataset Summary")
    print("-----------------------")
    print(f"Total image files: {len(records) + len(invalid_files)}")
    print(f"Valid records: {len(records)}")
    print(f"Invalid records: {len(invalid_files)}")

    print("\nAge statistics:")
    print(f"Minimum age: {min(ages)}")
    print(f"Maximum age: {max(ages)}")
    print(f"Average age: {sum(ages) / len(ages):.2f}")

    print("\nGender distribution:")
    print(f"Gender 0: {genders.count(0)}")
    print(f"Gender 1: {genders.count(1)}")

    print("\nEthnicity distribution:")
    for ethnicity in range(5):
        print(f"Ethnicity {ethnicity}: {ethnicities.count(ethnicity)}")

    if invalid_files:
        print("\nInvalid filenames:")
        for filename in invalid_files:
            print(filename)


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset folder not found: {DATASET_PATH}")
        return

    records, invalid_files = load_metadata(DATASET_PATH)
    display_dataset_summary(records, invalid_files)


if __name__ == "__main__":
    main()