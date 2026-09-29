import kagglehub
import shutil
import os

target_dir = r"C:\Users\kashy\.gemini\antigravity\scratch\CVDI\data"
os.makedirs(target_dir, exist_ok=True)

try:
    print("Downloading ECG Arrhythmia...")
    path2 = kagglehub.dataset_download("sadmansakib7/ecg-arrhythmia-classification-dataset")
    print("Downloaded to:", path2)
    # copy contents to target_dir
    for item in os.listdir(path2):
        s = os.path.join(path2, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
except Exception as e:
    print("Error downloading 2:", e)

try:
    print("Downloading Cardiovascular Disease...")
    path3 = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")
    print("Downloaded to:", path3)
    for item in os.listdir(path3):
        s = os.path.join(path3, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
except Exception as e:
    print("Error downloading 3:", e)
