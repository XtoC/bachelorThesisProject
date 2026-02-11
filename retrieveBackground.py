import os
meta_folder = "audio/background_audio/"

background_category = ["office", "cafe/restaurant", "residential_area", "city_center"]

def retrieve_file(file_type):
    for i in range(1,5):
        meta_file = os.path.join(meta_folder, f"fold{i}_{file_type}.txt")
        with open(meta_file) as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                if (parts[1] in background_category):
                    print(parts)

if __name__ == "__main__":
    retrieve_file("train")