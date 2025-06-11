import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import os
import threading
import random
import math
from eigenface_engine import EigenfaceEngine
from utils import load_and_validate_dataset

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎭 APLIKASI FACE RECOGNITION EIGENFACES")
        self.root.geometry("1400x900")
        self.root.configure(bg='#ecf0f1')
        
        # Variables
        self.dataset_folder = None
        self.test_image_path = None
        self.engine = EigenfaceEngine()
        self.is_trained = False
        self.dataset_images = []
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="🎭 APLIKASI FACE RECOGNITION EIGENFACES", 
                              font=('Arial', 18, 'bold'), 
                              fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(title_frame, 
                                 text="Cholif Bima Ardiansyah", 
                                 font=('Arial', 12), 
                                 fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Controls (Scrollable)
        left_container = tk.Frame(main_frame, bg='#ecf0f1')
        left_container.pack(side='left', fill='y', padx=(0, 15))
        
        # Create scrollable control panel
        control_canvas = tk.Canvas(left_container, bg='#f8f9fa', width=280)
        control_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=control_canvas.yview)
        left_frame = tk.Frame(control_canvas, bg='#f8f9fa')
        
        # Configure scrollable frame
        left_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        
        control_canvas.create_window((0, 0), window=left_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        
        # Pack scrollable components
        control_canvas.pack(side="left", fill="both", expand=True)
        control_scrollbar.pack(side="right", fill="y")
        
        # Add control panel title
        title_frame = tk.Frame(left_frame, bg='#2c3e50', height=40)
        title_frame.pack(fill='x', pady=(0, 10))
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="🎮 Control Panel", 
                font=('Arial', 12, 'bold'), 
                bg='#2c3e50', fg='#f8f9fa').pack(expand=True)
        
        # Bind mouse wheel to control panel
        def _on_control_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind mouse wheel events
        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", _on_control_mousewheel)
            for child in widget.winfo_children():
                bind_mousewheel(child)
        
        # Apply mouse wheel binding to all widgets in left frame
        left_frame.bind("<MouseWheel>", _on_control_mousewheel)
        
        # Store references for later use
        self.control_canvas = control_canvas
        self.bind_mousewheel = bind_mousewheel
        
        # Add padding container
        content_frame = tk.Frame(left_frame, bg='#f8f9fa')
        content_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Dataset section
        dataset_section = tk.LabelFrame(content_frame, text="📂 1. Dataset Training", 
                                       font=('Arial', 10, 'bold'),
                                       bg='#f8f9fa', fg='#2c3e50')
        dataset_section.pack(fill='x', pady=(0, 15))
        
        # Dataset buttons
        tk.Button(dataset_section, text="📁 Pilih Folder Dataset", 
                 command=self.select_dataset_folder,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                 relief='flat', padx=20, pady=8).pack(pady=5, fill='x')
        
        self.dataset_label = tk.Label(dataset_section, text="Dataset: -",
                                     font=('Arial', 9), fg='#7f8c8d', bg='#f8f9fa',
                                     wraplength=180)
        self.dataset_label.pack(pady=5)
        
        # Buttons row
        buttons_frame = tk.Frame(dataset_section, bg='#f8f9fa')
        buttons_frame.pack(fill='x', pady=5)
        
        self.train_btn = tk.Button(buttons_frame, text="🧠 Train", 
                                  command=self.train_model_threaded,
                                  bg='#e74c3c', fg='white', font=('Arial', 9, 'bold'),
                                  relief='flat', state='disabled')
        self.train_btn.pack(side='left', fill='x', expand=True, padx=(0, 2))
        
        self.reset_btn = tk.Button(buttons_frame, text="🔄 Reset", 
                                  command=self.reset_model,
                                  bg='#f39c12', fg='white', font=('Arial', 9, 'bold'),
                                  relief='flat', state='disabled')
        self.reset_btn.pack(side='left', fill='x', expand=True, padx=(2, 0))
        
        self.gallery_btn = tk.Button(dataset_section, text="🖼️ Lihat Dataset Gallery", 
                                    command=self.show_dataset_gallery,
                                    bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'),
                                    relief='flat', state='disabled')
        self.gallery_btn.pack(fill='x', pady=(5, 0))
        
        # Test section
        test_section = tk.LabelFrame(content_frame, text="🔍 2. Test Image", 
                                    font=('Arial', 10, 'bold'),
                                    bg='#f8f9fa', fg='#2c3e50')
        test_section.pack(fill='x', pady=(0, 15))
        
        tk.Button(test_section, text="📷 Pilih Gambar Test", 
                 command=self.select_test_image,
                 bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'),
                 relief='flat', padx=20, pady=8).pack(pady=5, fill='x')
        
        self.test_image_label = tk.Label(test_section, text="Test Image: -",
                                        font=('Arial', 9), fg='#7f8c8d', bg='#f8f9fa',
                                        wraplength=180)
        self.test_image_label.pack(pady=5)
        
        # Threshold section
        threshold_section = tk.LabelFrame(content_frame, text="⚙️ 3. Threshold Setting", 
                                         font=('Arial', 10, 'bold'),
                                         bg='#f8f9fa', fg='#2c3e50')
        threshold_section.pack(fill='x', pady=(0, 15))
        
        tk.Label(threshold_section, text="Threshold:", font=('Arial', 9), 
                bg='#f8f9fa').pack()
        
        self.threshold_var = tk.DoubleVar(value=0.8)
        self.threshold_label = tk.Label(threshold_section, text="0.8", 
                                       font=('Arial', 10, 'bold'), bg='#f8f9fa')
        self.threshold_label.pack()
        
        threshold_scale = tk.Scale(threshold_section, from_=0.1, to=2.0, resolution=0.1,
                                  orient='horizontal', variable=self.threshold_var,
                                  length=180, bg='#f8f9fa', 
                                  command=self.update_threshold_label)
        threshold_scale.pack(pady=5)
        
        # Recognition section
        recognition_section = tk.LabelFrame(content_frame, text="🎯 4. Face Recognition", 
                                          font=('Arial', 10, 'bold'),
                                          bg='#f8f9fa', fg='#2c3e50')
        recognition_section.pack(fill='x', pady=(0, 15))
        
        self.recognize_btn = tk.Button(recognition_section, text="🚀 Recognize Face", 
                                      command=self.recognize_face,
                                      bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                                      relief='flat', padx=20, pady=12, state='disabled')
        self.recognize_btn.pack(pady=5, fill='x')
        
        # Model info section
        info_section = tk.LabelFrame(content_frame, text="💾 5. Model Management", 
                                    font=('Arial', 10, 'bold'),
                                    bg='#f8f9fa', fg='#2c3e50')
        info_section.pack(fill='x')
        
        info_buttons = [
            ("📊 Model Info", self.show_model_info, "#34495e"),
            ("💾 Save Model", self.save_model, "#16a085"),
            ("📂 Load Model", self.load_model, "#8e44ad")
        ]
        
        for text, command, color in info_buttons:
            btn = tk.Button(info_section, text=text, command=command,
                           bg=color, fg='white', font=('Arial', 9),
                           relief='flat', padx=15, pady=5)
            btn.pack(pady=2, fill='x')
        
        # Apply mouse wheel binding to all widgets in control panel
        self.bind_mousewheel(left_frame)
        
        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg='#ecf0f1')
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Recognition Results (NEW INTERACTIVE)
        self.results_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.results_frame, text="🎯 Recognition Results")
        
        # Tab 2: Eigenfaces
        self.eigenfaces_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.eigenfaces_frame, text="👻 Eigenfaces")
        
        # Tab 3: Mean Face
        self.meanface_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.meanface_frame, text="👤 Mean Face")
        
        # Tab 4: Training Log
        self.log_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.log_frame, text="📝 Training Log")
        
        # Initialize displays
        self.init_displays()
        
    def init_displays(self):
        # Recognition Results Tab (Interactive)
        self.init_recognition_display()
        
        # Training Log Tab
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=20, width=60,
                                                 font=('Consolas', 10), bg='#2c3e50', fg='#ecf0f1')
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.log_message("🎭 FACE RECOGNITION EIGENFACES")
        self.log_message("=" * 50)
        self.log_message("📋 Langkah-langkah:")
        self.log_message("1️⃣ Pilih folder dataset")
        self.log_message("2️⃣ Train model")
        self.log_message("3️⃣ Pilih gambar test")  
        self.log_message("4️⃣ Recognize face")
        self.log_message("")
        
    def init_recognition_display(self):
        # Main recognition frame
        main_rec_frame = tk.Frame(self.results_frame, bg='white')
        main_rec_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title = tk.Label(main_rec_frame, text="🎯 Face Recognition Results", 
                        font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
        title.pack(pady=(0, 10))
        
        # Create scrollable frame
        canvas = tk.Canvas(main_rec_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_rec_frame, orient="vertical", command=canvas.yview)
        self.results_container = tk.Frame(canvas, bg='white')
        
        # Configure scrollable frame
        self.results_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.results_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store canvas reference for mouse wheel binding
        self.results_canvas = canvas
        
        # Bind mouse wheel to results canvas
        def _on_results_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind mouse wheel to results area only
        def bind_results_mousewheel(widget):
            widget.bind("<MouseWheel>", _on_results_mousewheel)
            for child in widget.winfo_children():
                bind_results_mousewheel(child)
        
        bind_results_mousewheel(self.results_container)
        
        # Initial message
        self.init_message = tk.Label(self.results_container, 
                                    text="👋 Ready for face recognition!\n\n"
                                         "1. Train your model first\n"
                                         "2. Select a test image\n" 
                                         "3. Click 'Recognize Face'\n"
                                         "4. See the magic happen! ✨",
                                    font=('Arial', 12), bg='white', fg='#7f8c8d',
                                    justify='center')
        self.init_message.pack(expand=True, pady=50)
        
    def update_threshold_label(self, value):
        self.threshold_label.config(text=f"{float(value):.1f}")
        
    def select_dataset_folder(self):
        folder = filedialog.askdirectory(title="Pilih Folder Dataset")
        if folder:
            self.dataset_folder = folder
            folder_name = os.path.basename(folder)
            self.dataset_label.config(text=f"📂 {folder_name}")
            
            # Validate and load dataset info
            try:
                images, labels, people = load_and_validate_dataset(folder)
                total_images = len(images)
                total_people = len(people)
                
                self.dataset_images = images  # Store for gallery
                
                self.log_message(f"✅ Dataset loaded: {folder_name}")
                self.log_message(f"👥 {total_people} people, 📸 {total_images} images")
                self.log_message(f"📋 People: {', '.join(people)}")
                
                # Enable buttons
                self.train_btn.config(state='normal')
                self.gallery_btn.config(state='normal')
                
            except Exception as e:
                self.log_message(f"❌ Dataset error: {str(e)}")
                messagebox.showerror("Error", f"Dataset tidak valid:\n{str(e)}")
                
    def select_test_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Test",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.test_image_path = file_path
            filename = os.path.basename(file_path)
            self.test_image_label.config(text=f"📷 {filename}")
            self.log_message(f"📷 Test image: {filename}")
            
            if self.is_trained:
                self.recognize_btn.config(state='normal')
                
    def train_model_threaded(self):
        threading.Thread(target=self.train_model, daemon=True).start()
        
    def train_model(self):
        if not self.dataset_folder:
            messagebox.showerror("Error", "Pilih folder dataset terlebih dahulu!")
            return
            
        try:
            self.train_btn.config(state='disabled', text='🔄 Training...')
            self.log_message("🧠 Memulai training model...")
            
            # Train model
            success = self.engine.train(self.dataset_folder)
            
            if success:
                self.is_trained = True
                self.log_message("✅ Training berhasil!")
                
                # Show model info
                info = self.engine.get_model_info()
                self.log_message(f"📊 Model Info:")
                self.log_message(f"   👥 Jumlah orang: {info['num_people']}")
                self.log_message(f"   📸 Total gambar: {info['total_images']}")
                self.log_message(f"   📐 Ukuran gambar: {info['image_size']}")
                self.log_message(f"   👻 Jumlah eigenfaces: {info['num_eigenfaces']}")
                
                # Enable buttons
                self.train_btn.config(state='normal', text='🧠 Train')
                self.reset_btn.config(state='normal')
                if self.test_image_path:
                    self.recognize_btn.config(state='normal')
                    
                # Update displays
                self.update_eigenfaces_display()
                self.update_meanface_display()
                
            else:
                self.log_message("❌ Training gagal!")
                self.train_btn.config(state='normal', text='🧠 Train')
                
        except Exception as e:
            self.log_message(f"❌ Training error: {str(e)}")
            self.train_btn.config(state='normal', text='🧠 Train')
            messagebox.showerror("Error", f"Training gagal:\n{str(e)}")
            
    def reset_model(self):
        if messagebox.askyesno("Konfirmasi", "Reset model training?\nAnda perlu train ulang."):
            self.engine = EigenfaceEngine()
            self.is_trained = False
            self.log_message("🔄 Model di-reset")
            
            # Reset buttons
            self.reset_btn.config(state='disabled')
            self.recognize_btn.config(state='disabled')
            
            # Clear displays
            for widget in self.eigenfaces_frame.winfo_children():
                widget.destroy()
            for widget in self.meanface_frame.winfo_children():
                widget.destroy()
                
            # Reset recognition display
            for widget in self.results_container.winfo_children():
                widget.destroy()
            
            # Restore initial message
            self.init_message = tk.Label(self.results_container, 
                                        text="👋 Ready for face recognition!\n\n"
                                             "1. Train your model first\n"
                                             "2. Select a test image\n" 
                                             "3. Click 'Recognize Face'\n"
                                             "4. See the magic happen! ✨",
                                        font=('Arial', 12), bg='white', fg='#7f8c8d',
                                        justify='center')
            self.init_message.pack(expand=True, pady=50)
            
    def show_dataset_gallery(self):
        if not self.dataset_images:
            messagebox.showinfo("Info", "Load dataset terlebih dahulu!")
            return
            
        # Create gallery window
        gallery_window = tk.Toplevel(self.root)
        gallery_window.title("🖼️ Dataset Gallery")
        gallery_window.geometry("1000x700")
        gallery_window.configure(bg='white')
        
        # Title
        title = tk.Label(gallery_window, text="🖼️ Dataset Training Images", 
                        font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
        title.pack(pady=10)
        
        # Scrollable frame
        canvas = tk.Canvas(gallery_window, bg='white')
        scrollbar = ttk.Scrollbar(gallery_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Group images by person
        people_images = {}
        for img_path in self.dataset_images:
            person = os.path.basename(os.path.dirname(img_path))
            if person not in people_images:
                people_images[person] = []
            people_images[person].append(img_path)
        
        # Display images
        row = 0
        for person, images in people_images.items():
            # Person header
            person_frame = tk.Frame(scrollable_frame, bg='#ecf0f1', relief='ridge', bd=2)
            person_frame.pack(fill='x', padx=10, pady=5)
            
            tk.Label(person_frame, text=f"👤 {person} ({len(images)} images)", 
                    font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack(pady=5)
            
            # Images grid
            images_frame = tk.Frame(scrollable_frame, bg='white')
            images_frame.pack(fill='x', padx=20, pady=5)
            
            for i, img_path in enumerate(images[:12]):  # Show max 12 per person
                try:
                    img = Image.open(img_path)
                    img = img.resize((80, 80), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    col = i % 6
                    if col == 0:
                        row_frame = tk.Frame(images_frame, bg='white')
                        row_frame.pack(fill='x', pady=2)
                    
                    img_label = tk.Label(row_frame, image=photo, bg='white', relief='solid', bd=1)
                    img_label.image = photo  # Keep reference
                    img_label.pack(side='left', padx=2)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            if len(images) > 12:
                tk.Label(images_frame, text=f"... dan {len(images)-12} gambar lainnya", 
                        font=('Arial', 9), bg='white', fg='#7f8c8d').pack(pady=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def recognize_face(self):
        if not self.is_trained:
            messagebox.showerror("Error", "Train model terlebih dahulu!")
            return
            
        if not self.test_image_path:
            messagebox.showerror("Error", "Pilih gambar test terlebih dahulu!")
            return
            
        try:
            self.recognize_btn.config(state='disabled', text='🔄 Recognizing...')
            
            # Recognize
            result = self.engine.recognize(self.test_image_path, self.threshold_var.get())
            
            # Update interactive display
            self.update_recognition_display(result)
            
            # Log result
            if result['recognized']:
                self.log_message(f"✅ Recognized: {result['person']} (distance: {result['distance']:.4f})")
            else:
                self.log_message(f"❌ Not recognized (distance: {result['distance']:.4f})")
                
            self.recognize_btn.config(state='normal', text='🚀 Recognize Face')
            
        except Exception as e:
            self.log_message(f"❌ Recognition error: {str(e)}")
            self.recognize_btn.config(state='normal', text='🚀 Recognize Face')
            messagebox.showerror("Error", f"Recognition gagal:\n{str(e)}")
            
    def update_recognition_display(self, result):
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()
            
        # Main result frame
        result_frame = tk.Frame(self.results_container, bg='white')
        result_frame.pack(fill='x', padx=20, pady=20)
        
        # Test image section
        test_section = tk.Frame(result_frame, bg='#f8f9fa', relief='ridge', bd=2)
        test_section.pack(fill='x', pady=(0, 20))
        
        tk.Label(test_section, text="📷 Test Image", font=('Arial', 14, 'bold'), 
                bg='#f8f9fa', fg='#2c3e50').pack(pady=10)
        
        # Load and display test image
        try:
            test_img = Image.open(self.test_image_path)
            test_img = test_img.resize((150, 150), Image.Resampling.LANCZOS)
            test_photo = ImageTk.PhotoImage(test_img)
            
            test_label = tk.Label(test_section, image=test_photo, bg='#f8f9fa')
            test_label.image = test_photo
            test_label.pack(pady=(0, 10))
            
            filename = os.path.basename(self.test_image_path)
            tk.Label(test_section, text=filename, font=('Arial', 10), 
                    bg='#f8f9fa', fg='#7f8c8d').pack(pady=(0, 10))
        except:
            tk.Label(test_section, text="❌ Error loading image", 
                    font=('Arial', 12), bg='#f8f9fa', fg='#e74c3c').pack(pady=20)
        
        # Result section
        if result['recognized']:
            # SUCCESS - Recognized
            result_section = tk.Frame(result_frame, bg='#d5f4e6', relief='ridge', bd=3)
            result_section.pack(fill='x', pady=(0, 20))
            
            tk.Label(result_section, text="✅ RECOGNIZED!", font=('Arial', 16, 'bold'), 
                    bg='#d5f4e6', fg='#27ae60').pack(pady=10)
                    
            person_frame = tk.Frame(result_section, bg='#d5f4e6')
            person_frame.pack(pady=10)
            
            tk.Label(person_frame, text=f"👤 Person: {result['person']}", 
                    font=('Arial', 14, 'bold'), bg='#d5f4e6', fg='#2c3e50').pack()
            tk.Label(person_frame, text=f"📏 Distance: {result['distance']:.4f}", 
                    font=('Arial', 12), bg='#d5f4e6', fg='#27ae60').pack()
            tk.Label(person_frame, text=f"🎯 Threshold: {self.threshold_var.get():.1f}", 
                    font=('Arial', 12), bg='#d5f4e6', fg='#7f8c8d').pack()
            
            # Show similar training images
            similar_section = tk.Frame(result_frame, bg='white')
            similar_section.pack(fill='x')
            
            tk.Label(similar_section, text="🔍 Similar Training Images", 
                    font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50').pack(pady=(0, 10))
            
            # Find images of recognized person
            person_images = []
            for img_path in self.dataset_images:
                if os.path.basename(os.path.dirname(img_path)) == result['person']:
                    person_images.append(img_path)
            
            # Show some random samples
            samples = random.sample(person_images, min(4, len(person_images)))
            
            images_frame = tk.Frame(similar_section, bg='white')
            images_frame.pack()
            
            for i, img_path in enumerate(samples):
                try:
                    img = Image.open(img_path)
                    img = img.resize((100, 100), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    img_label = tk.Label(images_frame, image=photo, bg='white', 
                                       relief='solid', bd=2)
                    img_label.image = photo
                    img_label.grid(row=0, column=i, padx=5, pady=5)
                    
                    # Filename
                    filename = os.path.basename(img_path)
                    tk.Label(images_frame, text=filename[:10]+"...", 
                            font=('Arial', 8), bg='white', fg='#7f8c8d').grid(row=1, column=i)
                except:
                    pass
        else:
            # FAILED - Not recognized
            result_section = tk.Frame(result_frame, bg='#ffeaa7', relief='ridge', bd=3)
            result_section.pack(fill='x', pady=(0, 20))
            
            tk.Label(result_section, text="❌ NOT RECOGNIZED", font=('Arial', 16, 'bold'), 
                    bg='#ffeaa7', fg='#e17055').pack(pady=10)
                    
            info_frame = tk.Frame(result_section, bg='#ffeaa7')
            info_frame.pack(pady=10)
            
            tk.Label(info_frame, text=f"📏 Distance: {result['distance']:.4f}", 
                    font=('Arial', 12), bg='#ffeaa7', fg='#2c3e50').pack()
            tk.Label(info_frame, text=f"🎯 Threshold: {self.threshold_var.get():.1f}", 
                    font=('Arial', 12), bg='#ffeaa7', fg='#7f8c8d').pack()
            tk.Label(info_frame, text="💡 Tip: Coba adjust threshold atau gunakan gambar yang ada di dataset", 
                    font=('Arial', 10), bg='#ffeaa7', fg='#6c5ce7').pack(pady=5)
        
        # Add bottom padding for better scrolling
        bottom_padding = tk.Frame(self.results_container, bg='white', height=50)
        bottom_padding.pack(fill='x')
            
    def show_model_info(self):
        if not self.is_trained:
            messagebox.showinfo("Info", "Train model terlebih dahulu!")
            return
            
        info = self.engine.get_model_info()
        info_text = f"""
📊 MODEL INFORMATION

👥 Number of People: {info['num_people']}
📸 Total Images: {info['total_images']}
📐 Image Size: {info['image_size']}
👻 Number of Eigenfaces: {info['num_eigenfaces']}

🧮 Top 10 Eigenvalues:
"""
        for i, val in enumerate(info['eigenvalues'][:10], 1):
            info_text += f"   {i:2d}. {val:.2f}\n"
            
        messagebox.showinfo("Model Info", info_text)
        
    def save_model(self):
        if not self.is_trained:
            messagebox.showinfo("Info", "Train model terlebih dahulu!")
            return
            
        try:
            self.engine.save_model("face_recognition_model.pkl")
            self.log_message("💾 Model saved: face_recognition_model.pkl")
            messagebox.showinfo("Success", "Model berhasil disimpan!")
        except Exception as e:
            self.log_message(f"❌ Save error: {str(e)}")
            messagebox.showerror("Error", f"Gagal menyimpan model:\n{str(e)}")
            
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl")]
        )
        if file_path:
            try:
                self.engine.load_model(file_path)
                self.is_trained = True
                self.log_message(f"📂 Model loaded: {os.path.basename(file_path)}")
                
                # Enable buttons
                self.reset_btn.config(state='normal')
                if self.test_image_path:
                    self.recognize_btn.config(state='normal')
                    
                # Update displays
                self.update_eigenfaces_display()
                self.update_meanface_display()
                
                messagebox.showinfo("Success", "Model berhasil dimuat!")
            except Exception as e:
                self.log_message(f"❌ Load error: {str(e)}")
                messagebox.showerror("Error", f"Gagal memuat model:\n{str(e)}")
                
    def update_eigenfaces_display(self):
        # Clear previous
        for widget in self.eigenfaces_frame.winfo_children():
            widget.destroy()
            
        try:
            eigenfaces = self.engine.get_eigenfaces()
            if eigenfaces is None:
                return
                
            # Title
            title = tk.Label(self.eigenfaces_frame, text="👻 Top 10 Eigenfaces", 
                            font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
            title.pack(pady=10)
            
            # Main container
            container = tk.Frame(self.eigenfaces_frame, bg='white')
            container.pack(fill='both', expand=True, padx=20, pady=10)
            
            # Display eigenfaces in grid
            for i in range(min(10, eigenfaces.shape[1])):
                row = i // 5
                col = i % 5
                
                # Create frame for each eigenface
                eigen_frame = tk.Frame(container, bg='white', relief='ridge', bd=1)
                eigen_frame.grid(row=row*2, column=col, padx=5, pady=5)
                
                # Get eigenface and normalize
                eigenface = eigenfaces[:, i].reshape(64, 64)
                eigenface = ((eigenface - eigenface.min()) / (eigenface.max() - eigenface.min()) * 255).astype(np.uint8)
                
                # Convert to image
                img = Image.fromarray(eigenface, mode='L')
                img = img.resize((100, 100), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Display
                img_label = tk.Label(eigen_frame, image=photo, bg='white')
                img_label.image = photo
                img_label.pack()
                
                # Label
                tk.Label(eigen_frame, text=f"Eigenface {i+1}", 
                        font=('Arial', 9, 'bold'), bg='white').pack()
                        
        except Exception as e:
            tk.Label(self.eigenfaces_frame, text=f"❌ Error: {str(e)}", 
                    font=('Arial', 12), bg='white', fg='#e74c3c').pack(expand=True)
            
    def update_meanface_display(self):
        # Clear previous
        for widget in self.meanface_frame.winfo_children():
            widget.destroy()
            
        try:
            mean_face = self.engine.get_mean_face()
            if mean_face is None:
                return
                
            # Title
            title = tk.Label(self.meanface_frame, text="👤 Mean Face (Average Face)", 
                            font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
            title.pack(pady=20)
            
            # Reshape and normalize
            mean_face_img = mean_face.reshape(64, 64)
            mean_face_img = ((mean_face_img - mean_face_img.min()) / 
                           (mean_face_img.max() - mean_face_img.min()) * 255).astype(np.uint8)
            
            # Convert to image
            img = Image.fromarray(mean_face_img, mode='L')
            img = img.resize((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Display
            img_label = tk.Label(self.meanface_frame, image=photo, bg='white', relief='ridge', bd=2)
            img_label.image = photo
            img_label.pack(expand=True)
            
            # Description
            desc = tk.Label(self.meanface_frame, 
                           text="📝 Rata-rata dari semua gambar training\n"
                                "Digunakan untuk normalisasi sebelum PCA", 
                           font=('Arial', 12), bg='white', fg='#7f8c8d', justify='center')
            desc.pack(pady=20)
            
        except Exception as e:
            tk.Label(self.meanface_frame, text=f"❌ Error: {str(e)}", 
                    font=('Arial', 12), bg='white', fg='#e74c3c').pack(expand=True)
            
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 