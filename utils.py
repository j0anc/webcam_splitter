import os
import shutil


def create_output_dir(output_dir: str) -> None:
    if os.path.exists(output_dir):
        print(f"Directory '{output_dir}' already exists.")
    else:
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    return


def clear_create_dir(dir: str) -> None:
    if os.path.exists(dir):
        print(f"Existing files in {dir} will be cleared...")
        shutil.rmtree(dir)

    os.makedirs(dir)
    print(f"Directory '{dir}' created.")
    return
