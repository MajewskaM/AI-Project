import os

def rearrange_files(source_directory, destination_directory, desired_file_name):

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Iterate over the files in the source directory
    for filename in os.listdir(source_directory):
        # Check if the file is a text file
        if filename.endswith(desired_file_name):
            parse_file(os.path.join(source_directory, filename), destination_directory)

    print("Data from file: " + str(desired_file_name) + " - rearranged successfully.")

    
# Function to parse each file and extract class information
def parse_file(file_path, destination_directory):
    with open(file_path, 'r') as file:
        num_examples = int(file.readline())
        num_pixels = int(file.readline())
        prev_class = -1
        index = 0
        for _ in range(num_examples):
            example = list(map(float, file.readline().split()))
            class_number = int(example[num_pixels + 2])
           
            if class_number != prev_class:
                index = 1
                prev_class = class_number
            else:
                index += 1
            features = example[:num_pixels]

            class_directory = os.path.join(destination_directory, str(class_number))
            if not os.path.exists(class_directory):
                os.makedirs(class_directory)

            # write the features to a new file
            output_file_path = os.path.join(class_directory, f"{index}.txt")
            with open(output_file_path, 'w') as output_file:
                output_file.write(" ".join(map(str, features)))
