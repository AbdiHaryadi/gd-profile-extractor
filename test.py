import csv
from extract_profile_data import extract_profile_data

if __name__ == "__main__":
    ss_dir = "dataset/screenshots/"

    profile_map = {}
    with open("dataset/screenshots_label.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            profile = {}
            path = ss_dir + row["filename"]
            for key, value in row.items():
                if key != "filename":
                    if key == "name":
                        profile[key] = value
                    else:
                        profile[key] = int(value)

            profile_map[path] = profile

    for path, profile in profile_map.items():
        prediction = extract_profile_data(path)

        print(path)
        print(prediction)
        print()

        for key, value in prediction.items():
            expected_data = profile[key]
            actual_data = prediction[key]
            
            if expected_data != actual_data:
                if type(expected_data) == str:
                    if expected_data.lower() != actual_data.lower():
                        print("[case insensitive] {} - {}: Expected {}, got {}".format(path, key, expected_data, actual_data))
                else:
                    print("{} - {}: Expected {}, got {}".format(path, key, expected_data, actual_data))
