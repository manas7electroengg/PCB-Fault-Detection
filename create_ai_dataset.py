import os
import cv2
import shutil

# ============================================================
# PCB VISION - AI DEFECT DATASET CREATION
# ============================================================

DATASET_PATH = "DeepPCB-master/DeepPCB-master/PCBData"
OUTPUT_PATH = "ai_defect_dataset"

# DeepPCB defect classes
DEFECT_CLASSES = {
    1: "Open",
    2: "Short",
    3: "Mousebite",
    4: "Spur",
    5: "Copper",
    6: "Pin-hole"
}

# ------------------------------------------------------------
# DELETE OLD DATASET IF IT EXISTS
# ------------------------------------------------------------

if os.path.exists(OUTPUT_PATH):

    print("Removing old AI dataset...")

    shutil.rmtree(OUTPUT_PATH)


# ------------------------------------------------------------
# CREATE FOLDERS
# ------------------------------------------------------------

for dataset_type in ["train", "validation"]:

    for defect_name in DEFECT_CLASSES.values():

        folder_path = os.path.join(
            OUTPUT_PATH,
            dataset_type,
            defect_name
        )

        os.makedirs(folder_path, exist_ok=True)


print("=" * 60)
print("PCB VISION - AI DEFECT DATASET CREATION")
print("=" * 60)

print("\nCreating defect crops...\n")


total_samples = 0

# ------------------------------------------------------------
# PROCESS ALL GROUPS
# ------------------------------------------------------------

groups = sorted(os.listdir(DATASET_PATH))

for group in groups:

    group_path = os.path.join(DATASET_PATH, group)

    if not os.path.isdir(group_path):
        continue


    # Find annotation folders
    for folder in os.listdir(group_path):

        if not folder.endswith("_not"):
            continue


        annotation_folder = os.path.join(
            group_path,
            folder
        )


        image_folder_name = folder.replace(
            "_not",
            ""
        )

        image_folder = os.path.join(
            group_path,
            image_folder_name
        )


        if not os.path.exists(image_folder):
            continue


        # ----------------------------------------------------
        # READ ANNOTATION FILES
        # ----------------------------------------------------

        for annotation_file in os.listdir(annotation_folder):

            if not annotation_file.endswith(".txt"):
                continue


            annotation_path = os.path.join(
                annotation_folder,
                annotation_file
            )


            image_id = annotation_file.replace(
                ".txt",
                ""
            )


            test_image_path = os.path.join(
                image_folder,
                image_id + "_test.jpg"
            )


            # Check image exists
            if not os.path.exists(test_image_path):
                continue


            # Read PCB image
            image = cv2.imread(test_image_path)


            if image is None:
                continue


            height, width = image.shape[:2]


            # Read annotation
            with open(annotation_path, "r") as file:

                lines = file.readlines()


            # ------------------------------------------------
            # PROCESS EVERY DEFECT
            # ------------------------------------------------

            for defect_number, line in enumerate(lines):


                values = line.strip().split()


                if len(values) != 5:
                    continue


                try:

                    x1 = int(values[0])
                    y1 = int(values[1])
                    x2 = int(values[2])
                    y2 = int(values[3])

                    class_id = int(values[4])

                except ValueError:

                    continue


                if class_id not in DEFECT_CLASSES:
                    continue


                # --------------------------------------------
                # FIX COORDINATES
                # --------------------------------------------

                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))

                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))


                # Ensure correct order
                if x1 > x2:
                    x1, x2 = x2, x1

                if y1 > y2:
                    y1, y2 = y2, y1


                # --------------------------------------------
                # ADD PADDING AROUND DEFECT
                # --------------------------------------------

                padding = 15

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)

                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)


                # Crop defect
                defect_crop = image[y1:y2, x1:x2]


                # Ignore invalid crop
                if defect_crop.size == 0:
                    continue


                defect_name = DEFECT_CLASSES[class_id]


                # --------------------------------------------
                # TRAIN / VALIDATION SPLIT
                # --------------------------------------------

                if total_samples % 5 == 0:

                    dataset_type = "validation"

                else:

                    dataset_type = "train"


                output_folder = os.path.join(
                    OUTPUT_PATH,
                    dataset_type,
                    defect_name
                )


                # Unique filename
                output_filename = (
                    f"{group}_"
                    f"{image_id}_"
                    f"{defect_number}.jpg"
                )


                output_path = os.path.join(
                    output_folder,
                    output_filename
                )


                # Save crop
                cv2.imwrite(
                    output_path,
                    defect_crop
                )


                total_samples += 1


print("\n" + "=" * 60)
print("AI DATASET CREATION COMPLETED")
print("=" * 60)

print("\nTotal defect samples:", total_samples)

print("\nDataset saved at:")
print(OUTPUT_PATH)

print("\nStructure:")

print("""
ai_defect_dataset/

├── train/
│   ├── Open/
│   ├── Short/
│   ├── Mousebite/
│   ├── Spur/
│   ├── Copper/
│   └── Pin-hole/

└── validation/
    ├── Open/
    ├── Short/
    ├── Mousebite/
    ├── Spur/
    ├── Copper/
    └── Pin-hole/
""")

print("Ready for AI model training!")