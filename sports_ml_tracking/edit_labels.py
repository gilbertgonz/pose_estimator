import os

def remove_class_1_annotations(file_path):
    # Read content from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filter out lines with class index 1
    lines = [line for line in lines if line.split()[0] != '1']

    # Write modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def replace_class_2_with_class_1(file_path):
    # Read content from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace class index 2 with class index 1
    lines = [line.replace('2 ', '1 ', 1) for line in lines]

    # Write modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def process_label_files(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            # Process label file
            # remove_class_1_annotations(file_path)
            replace_class_2_with_class_1(file_path)
            print(f"Processed: {file_path}")

# Directory containing label files
label_directory = '/home/gilberto/Downloads/basketball_dataset/train/labels'

# Process label files in the directory
process_label_files(label_directory)