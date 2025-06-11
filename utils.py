import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_images_from_folder(folder_path, target_size=(64, 64)):
    """
    Load semua gambar dari folder dan subfolder
    
    Args:
        folder_path (str): Path ke folder dataset
        target_size (tuple): Ukuran target gambar (width, height)
    
    Returns:
        images (list): List array gambar
        labels (list): List label untuk setiap gambar
        label_names (list): List nama unik labels
    """
    images = []
    labels = []
    label_names = []
    
    # Dapatkan semua subfolder
    subfolders = [f for f in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, f))]
    
    for i, subfolder in enumerate(subfolders):
        label_names.append(subfolder)
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # Load gambar dari setiap subfolder
        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(subfolder_path, filename)
                
                # Load dan preprocess gambar
                img = preprocess_image(img_path, target_size)
                if img is not None:
                    images.append(img)
                    labels.append(i)
    
    return np.array(images), np.array(labels), label_names

def preprocess_image(image_path, target_size=(64, 64)):
    """
    Preprocess gambar: grayscale, resize, normalize
    
    Args:
        image_path (str): Path ke file gambar
        target_size (tuple): Ukuran target gambar
    
    Returns:
        np.array: Array gambar yang sudah dipreprocess
    """
    try:
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert ke grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize gambar
        img_resized = cv2.resize(img_gray, target_size)
        
        # Normalize ke range [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def preprocess_single_image(image_path, target_size=(64, 64)):
    """
    Preprocess single gambar untuk testing
    
    Args:
        image_path (str): Path ke file gambar
        target_size (tuple): Ukuran target gambar
    
    Returns:
        np.array: Array gambar yang sudah dipreprocess dan flattened
    """
    img = preprocess_image(image_path, target_size)
    if img is not None:
        return img.flatten()
    return None

def create_folder_if_not_exists(folder_path):
    """
    Buat folder jika belum ada
    
    Args:
        folder_path (str): Path folder yang akan dibuat
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_eigenfaces_visualization(eigenfaces, save_path, n_components=10, img_shape=(64, 64)):
    """
    Simpan visualisasi eigenfaces
    
    Args:
        eigenfaces (np.array): Array eigenfaces
        save_path (str): Path untuk menyimpan visualisasi
        n_components (int): Jumlah eigenfaces yang akan ditampilkan
        img_shape (tuple): Bentuk gambar asli
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Top 10 Eigenfaces', fontsize=16)
    
    for i in range(min(n_components, eigenfaces.shape[0])):
        # Reshape eigenface ke bentuk gambar
        eigenface = eigenfaces[i].reshape(img_shape)
        
        # Normalize untuk visualisasi
        eigenface_normalized = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
        
        # Plot
        row, col = i // 5, i % 5
        axes[row, col].imshow(eigenface_normalized, cmap='gray')
        axes[row, col].set_title(f'Eigenface {i+1}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_components, 10):
        row, col = i // 5, i % 5
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_mean_face_visualization(mean_face, save_path, img_shape=(64, 64)):
    """
    Simpan visualisasi mean face
    
    Args:
        mean_face (np.array): Array mean face
        save_path (str): Path untuk menyimpan visualisasi
        img_shape (tuple): Bentuk gambar asli
    """
    # Reshape ke bentuk gambar
    mean_face_img = mean_face.reshape(img_shape)
    
    # Normalize untuk visualisasi
    mean_face_normalized = (mean_face_img - mean_face_img.min()) / (mean_face_img.max() - mean_face_img.min())
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(mean_face_normalized, cmap='gray')
    plt.title('Mean Face', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def validate_dataset_folder(folder_path):
    """
    Validasi struktur folder dataset
    
    Args:
        folder_path (str): Path ke folder dataset
    
    Returns:
        bool: True jika valid, False jika tidak
        str: Pesan error jika ada
    """
    if not os.path.exists(folder_path):
        return False, "Folder tidak ditemukan"
    
    subfolders = [f for f in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, f))]
    
    if len(subfolders) < 2:
        return False, "Dataset harus memiliki minimal 2 folder (2 orang berbeda)"
    
    total_images = 0
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        images = [f for f in os.listdir(subfolder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if len(images) < 1:
            return False, f"Folder {subfolder} tidak memiliki gambar yang valid"
        
        total_images += len(images)
    
    if total_images < 10:
        return False, "Dataset harus memiliki minimal 10 gambar total"
    
    return True, "Dataset valid"

def euclidean_distance(vec1, vec2):
    """
    Hitung euclidean distance antara dua vektor
    
    Args:
        vec1, vec2 (np.array): Vektor input
    
    Returns:
        float: Euclidean distance
    """
    return np.linalg.norm(vec1 - vec2)

def load_and_validate_dataset(folder_path):
    """
    Load dan validasi dataset
    
    Args:
        folder_path (str): Path ke folder dataset
    
    Returns:
        images (list): List path gambar
        labels (list): List label untuk setiap gambar  
        people (list): List nama orang
    """
    if not os.path.exists(folder_path):
        raise ValueError("Folder dataset tidak ditemukan")
    
    subfolders = [f for f in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, f))]
    
    if len(subfolders) < 2:
        raise ValueError("Dataset harus memiliki minimal 2 folder (2 orang berbeda)")
    
    images = []
    labels = []
    people = subfolders
    
    for i, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(folder_path, subfolder)
        img_files = [f for f in os.listdir(subfolder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if len(img_files) < 1:
            raise ValueError(f"Folder {subfolder} tidak memiliki gambar yang valid")
        
        for img_file in img_files:
            images.append(os.path.join(subfolder_path, img_file))
            labels.append(i)
    
    if len(images) < 10:
        raise ValueError("Dataset harus memiliki minimal 10 gambar total")
    
    return images, labels, people

def print_training_info(num_images, num_people, img_shape, num_components):
    """
    Print informasi training
    """
    print("="*50)
    print("TRAINING INFORMATION")
    print("="*50)
    print(f"Total gambar: {num_images}")
    print(f"Jumlah orang: {num_people}")
    print(f"Ukuran gambar: {img_shape[0]}x{img_shape[1]}")
    print(f"Jumlah eigenfaces: {num_components}")
    print("="*50) 