import os

def clear_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):  # Check if it's a file or a link, not a directory
            try:
                os.unlink(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

folders = [
    "C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\model_testing_text\\training_models\\saved_weights",
    "C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\model_testing_text\\training_models\\training_progress"
]

for folder in folders:
    clear_files_in_folder(folder)


