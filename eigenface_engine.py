import numpy as np
import joblib
import os
from utils import *

class EigenfaceEngine:
    def __init__(self, target_size=(64, 64), n_components=50):
        self.target_size = target_size
        self.n_components = n_components
        self.is_trained = False
        
        # Model components
        self.mean_face = None
        self.eigenfaces = None
        self.eigenvalues = None
        self.projected_faces = None
        self.face_labels = None
        self.label_names = None
        
    def _manual_eigenvalue_decomposition(self, cov_matrix):
        print("Menghitung eigenvalues dan eigenvectors secara manual...")
        n = cov_matrix.shape[0]
        
        eigenvalues = []
        eigenvectors = []
        
        # Implementasi manual power iteration untuk mendapatkan eigenvalue/eigenvector
        for i in range(min(self.n_components, n)):
            # Power iteration untuk eigenvalue terbesar
            v = np.random.rand(n)
            v = v / np.linalg.norm(v)
            
            # Deflasi untuk eigenvalue sebelumnya
            A_deflated = cov_matrix.copy()
            for j in range(len(eigenvalues)):
                A_deflated = A_deflated - eigenvalues[j] * np.outer(eigenvectors[j], eigenvectors[j])
            
            # Power iteration
            for iteration in range(100):  # Max 100 iterasi
                v_new = A_deflated @ v
                eigenvalue = np.dot(v, v_new)
                
                if np.linalg.norm(v_new) < 1e-10:
                    break
                    
                v_new = v_new / np.linalg.norm(v_new)
                
                # Check konvergensi
                if np.linalg.norm(v_new - v) < 1e-6:
                    break
                    
                v = v_new
            
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            
            print(f"Eigenvalue {i+1}: {eigenvalue:.6f}")
        
        return np.array(eigenvalues), np.array(eigenvectors).T
    
    def _alternative_eigendecomposition(self, A):
        print("Menggunakan trick Turk & Pentland untuk efisiensi...")
        
        # Hitung L = A^T * A (ukuran M x M)
        L = A.T @ A
        
        # Manual eigendecomposition untuk matriks L yang lebih kecil
        eigenvals_L, eigenvecs_L = self._manual_eigenvalue_decomposition(L)
        
        # Mapping kembali ke ruang asli: eigenvectors_original = A * eigenvecs_L
        eigenvectors_original = A @ eigenvecs_L
        
        # Normalisasi eigenvectors
        for i in range(eigenvectors_original.shape[1]):
            norm = np.linalg.norm(eigenvectors_original[:, i])
            if norm > 0:
                eigenvectors_original[:, i] = eigenvectors_original[:, i] / norm
        
        # Sort berdasarkan eigenvalue (descending)
        sorted_indices = np.argsort(eigenvals_L)[::-1]
        eigenvalues = eigenvals_L[sorted_indices]
        eigenvectors = eigenvectors_original[:, sorted_indices]
        
        return eigenvalues, eigenvectors
    
    def train(self, dataset_folder):
        try:
            print("Memulai training Eigenface model...")
            
            # Validasi dataset
            is_valid, error_msg = validate_dataset_folder(dataset_folder)
            if not is_valid:
                print(f"Error: {error_msg}")
                return False
            
            # Load images dari dataset
            print("Loading dataset...")
            images, labels, label_names = load_images_from_folder(dataset_folder, self.target_size)
            
            if len(images) == 0:
                print("Error: Tidak ada gambar yang berhasil di-load")
                return False
            
            self.label_names = label_names
            self.face_labels = labels
            
            # Print info training
            print_training_info(len(images), len(label_names), self.target_size, self.n_components)
            
            # Langkah 1: Konversi gambar ke vektor dan normalize
            print("Langkah 1: Konversi gambar ke vektor...")
            face_vectors = []
            for img in images:
                face_vectors.append(img.flatten())
            face_vectors = np.array(face_vectors)
            
            print(f"Shape face vectors: {face_vectors.shape}")
            
            # Langkah 2: Hitung rata-rata wajah (mean face)
            print("Langkah 2: Menghitung mean face...")
            self.mean_face = np.mean(face_vectors, axis=0)
            
            # Langkah 3: Kurangi mean face dari setiap gambar
            print("Langkah 3: Mean centering...")
            mean_centered_faces = face_vectors - self.mean_face
            
            # Langkah 4: Hitung eigenvalue dan eigenvector secara manual
            print("Langkah 4: Menghitung eigenvalues dan eigenvectors...")
            
            # Pilih metode berdasarkan ukuran data
            M, N = mean_centered_faces.shape  # M = jumlah gambar, N = dimensi pixel
            
            if M < N:
                # Gunakan trick Turk & Pentland
                eigenvalues, eigenvectors = self._alternative_eigendecomposition(mean_centered_faces.T)
            else:
                # Metode konvensional
                cov_matrix = np.cov(mean_centered_faces.T)
                eigenvalues, eigenvectors = self._manual_eigenvalue_decomposition(cov_matrix)
            
            # Ambil top-k eigenfaces
            self.eigenvalues = eigenvalues[:self.n_components]
            self.eigenfaces = eigenvectors[:, :self.n_components]
            
            print(f"Eigenfaces shape: {self.eigenfaces.shape}")
            
            # Langkah 5: Proyeksikan semua wajah training ke eigenface space
            print("Langkah 5: Proyeksi wajah ke eigenspace...")
            self.projected_faces = mean_centered_faces @ self.eigenfaces
            
            print(f"Projected faces shape: {self.projected_faces.shape}")
            
            # Set flag training selesai
            self.is_trained = True
            
            print("Training selesai!")
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False
    
    def recognize(self, test_image_path, threshold=0.8):
        if not self.is_trained:
            return {"status": "error", "message": "Model belum di-training"}
        
        try:
            # Preprocess gambar test
            test_face = preprocess_single_image(test_image_path, self.target_size)
            if test_face is None:
                return {"status": "error", "message": "Gagal memproses gambar test"}
            
            # Mean centering
            test_face_centered = test_face - self.mean_face
            
            # Proyeksi ke eigenspace
            test_projection = test_face_centered @ self.eigenfaces
            
            # Hitung jarak euclidean dengan semua wajah training
            distances = []
            for i, projected_face in enumerate(self.projected_faces):
                distance = euclidean_distance(test_projection, projected_face)
                distances.append({
                    'distance': distance,
                    'label': self.face_labels[i],
                    'name': self.label_names[self.face_labels[i]]
                })
            
            # Sort berdasarkan jarak terdekat
            distances.sort(key=lambda x: x['distance'])
            
            # Ambil hasil terdekat
            best_match = distances[0]
            
            # Cek threshold
            if best_match['distance'] < threshold:
                recognized = True
                message = f"Wajah dikenali sebagai: {best_match['name']}"
            else:
                recognized = False
                message = "Wajah tidak dikenali dalam database"
            
            return {
                "recognized": recognized,
                "person": best_match['name'] if recognized else "Unknown",
                "distance": best_match['distance'],
                "message": message,
                "all_distances": distances[:5],  # Top 5 hasil
                "threshold_used": threshold
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error during recognition: {str(e)}"}
    
    def save_model(self, save_path):
        if not self.is_trained:
            print("Model belum di-training")
            return False
        
        try:
            create_folder_if_not_exists(os.path.dirname(save_path))
            
            model_data = {
                'mean_face': self.mean_face,
                'eigenfaces': self.eigenfaces,
                'eigenvalues': self.eigenvalues,
                'projected_faces': self.projected_faces,
                'face_labels': self.face_labels,
                'label_names': self.label_names,
                'target_size': self.target_size,
                'n_components': self.n_components
            }
            
            joblib.dump(model_data, save_path)
            print(f"Model disimpan ke: {save_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                print("File model tidak ditemukan")
                return False
            
            model_data = joblib.load(model_path)
            
            self.mean_face = model_data['mean_face']
            self.eigenfaces = model_data['eigenfaces']
            self.eigenvalues = model_data['eigenvalues']
            self.projected_faces = model_data['projected_faces']
            self.face_labels = model_data['face_labels']
            self.label_names = model_data['label_names']
            self.target_size = model_data['target_size']
            self.n_components = model_data['n_components']
            
            self.is_trained = True
            print(f"Model berhasil di-load dari: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def save_visualizations(self, output_folder):
        if not self.is_trained:
            print("Model belum di-training")
            return
        
        create_folder_if_not_exists(output_folder)
        
        # Simpan visualisasi eigenfaces
        eigenfaces_path = os.path.join(output_folder, "eigenfaces.png")
        save_eigenfaces_visualization(self.eigenfaces.T, eigenfaces_path, 
                                      n_components=10, img_shape=self.target_size)
        
        # Simpan visualisasi mean face
        mean_face_path = os.path.join(output_folder, "mean_face.png")
        save_mean_face_visualization(self.mean_face, mean_face_path, 
                                     img_shape=self.target_size)
        
        print(f"Visualisasi disimpan di: {output_folder}")
    
    def get_model_info(self):
        if not self.is_trained:
            return {"status": "Model belum di-training"}
        
        return {
            "status": "Model sudah di-training",
            "num_people": len(self.label_names),
            "people_names": self.label_names,
            "total_images": len(self.face_labels),
            "image_size": f"{self.target_size[0]}x{self.target_size[1]}",
            "num_eigenfaces": self.n_components,
            "eigenvalues": self.eigenvalues.tolist() if hasattr(self, 'eigenvalues') else []
        }
    
    def get_eigenfaces(self):
        """Return eigenfaces array untuk visualisasi"""
        if not self.is_trained:
            return None
        return self.eigenfaces
    
    def get_mean_face(self):
        """Return mean face array untuk visualisasi"""
        if not self.is_trained:
            return None
        return self.mean_face
