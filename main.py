from extract_profile_data import extract_profile_data

if __name__ == "__main__":
    path = input("Path: ")
    result = extract_profile_data(path)
    print("---")
    for key, value in result.items():
        print("{}: {}".format(key, value))
    print("---")
    
