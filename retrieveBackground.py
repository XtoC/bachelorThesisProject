import os
import shutil
meta_folder = "audio/background_audio/"

background_category = ["office", "cafe/restaurant", "residential_area", "city_center"]

def retrieve_file(file_type):
    folder_type = file_type
    if (file_type == "evaluate"):
        folder_type = "test"

    for i in range(1,5):
        meta_file = os.path.join(meta_folder, f"fold{i}_{file_type}.txt")
        with open(meta_file) as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                if (parts[1] in background_category):
                    if (parts[1] == "cafe/restaurant"):
                        parts[1] = "cafe"
                    destination = os.path.join(meta_folder, f"{folder_type}_audio/{parts[1]}/")
                    shutil.copy(parts[0], destination)

if __name__ == "__main__":
    retrieve_file("train")
    retrieve_file("evaluate")