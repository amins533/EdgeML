import os
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FastModernCardViewer(ctk.CTk):

    def __init__(self, target_folder="cards"):
        super().__init__()

        self.title("High-Res Cards Viewer (Optimized)")
        self.geometry("1300x800")

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        self.current_dir = os.path.abspath(target_folder)
        self.raw_pil_image = None
        self.selected_file_path = None
        self.resize_after_id = None  # Debounce timer for smooth resizing

        # --- Top Navigation Bar ---
        self.top_frame = ctk.CTkFrame(self, corner_radius=10)
        self.top_frame.pack(fill=tk.X, padx=15, pady=15)

        self.up_btn = ctk.CTkButton(
            self.top_frame,
            text="📁 Up",
            width=90,
            font=("Segoe UI", 13, "bold"),
            command=self.go_up,
        )
        self.up_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.path_entry = ctk.CTkEntry(self.top_frame, font=("Segoe UI", 12))
        self.path_entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10
        )
        self.path_entry.insert(0, self.current_dir)
        self.path_entry.bind("<Return>", self.go_to_entered_path)

        self.pop_btn = ctk.CTkButton(
            self.top_frame,
            text="View Full Size",
            width=150,
            font=("Segoe UI", 13, "bold"),
            command=self.open_full_resolution,
            state="disabled",
        )
        self.pop_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        # --- Main Layout Split Panel ---
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Left Column: File Explorer List
        self.left_column = ctk.CTkFrame(self.main_container, width=350)
        self.left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        self.left_column.pack_propagate(False)

        self.listbox = tk.Listbox(
            self.left_column,
            font=("Segoe UI", 11),
            bg="#2b2b2b",
            fg="#ffffff",
            selectbackground="#1f538d",
            selectforeground="#ffffff",
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
        )
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right Column: Big Preview Box
        self.preview_frame = ctk.CTkScrollableFrame(
            self.main_container, label_text=" CARD PREVIEW (Double-click to zoom) "
        )
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.preview_label = ctk.CTkLabel(
            self.preview_frame, text="Select a PNG file to view it", font=("Segoe UI", 13)
        )
        self.preview_label.pack(expand=True, pady=40)

        # --- Event Bindings ---
        self.listbox.bind("<<ListboxSelect>>", self.on_select_item)
        self.listbox.bind("<Double-Button-1>", self.on_list_double_click)
        self.preview_label.bind("<Double-Button-1>", lambda e: self.open_full_resolution())
        
        # Optimized layout configuration listener
        self.preview_frame.bind("<Configure>", self.on_window_resize)

        self.load_directory(self.current_dir)

    def load_directory(self, path):
        try:
            self.listbox.delete(0, tk.END)
            self.current_dir = path
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, path)

            self.clear_preview()

            folders = []
            files = []

            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                try:
                    if os.path.isdir(full_path):
                        folders.append(f"📁  {item}")
                    elif item.lower().endswith(".png"):
                        files.append(f"🖼️  {item}")
                    else:
                        files.append(f"📄  {item}")
                except PermissionError:
                    continue

            for folder in sorted(folders):
                self.listbox.insert(tk.END, folder)
            for file in sorted(files):
                self.listbox.insert(tk.END, file)

        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder.\nDetails: {e}")

    def on_select_item(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return

        selected_item = self.listbox.get(selection[0])
        parts = selected_item.split("  ", 1)
        if len(parts) < 2:
            return
        clean_name = parts[1]

        target_path = os.path.join(self.current_dir, clean_name)

        if os.path.isfile(target_path) and clean_name.lower().endswith(".png"):
            self.selected_file_path = target_path
            try:
                # OPTIMIZATION 1: Use draft mode to load a lightweight version for the sidebar view
                img = Image.open(target_path)
                max_w = max(self.preview_frame.winfo_width() - 40, 300)
                max_h = max(self.preview_frame.winfo_height() - 40, 300)
                img.draft("RGB", (max_w, max_h))
                
                self.raw_pil_image = img
                self.pop_btn.configure(state="normal")
                self.update_preview_resolution(fast=True)
            except Exception as e:
                self.preview_label.configure(image="", text=f"Error loading image.\n{e}")
        else:
            self.clear_preview()

    def update_preview_resolution(self, fast=False):
        if self.raw_pil_image is None:
            return
        try:
            img_copy = self.raw_pil_image.copy()
            max_w = max(self.preview_frame.winfo_width() - 40, 300)
            max_h = max(self.preview_frame.winfo_height() - 40, 300)
            
            # OPTIMIZATION 2: Use rapid BILINEAR processing while browsing, use heavy LANCZOS only when stable
            resample_algo = Image.Resampling.BILINEAR if fast else Image.Resampling.LANCZOS
            img_copy.thumbnail((max_w, max_h), resample_algo)
            
            ctk_img = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=(img_copy.width, img_copy.height))
            self.preview_label.configure(image=ctk_img, text="")
        except Exception:
            pass

    def on_window_resize(self, event):
        """Prevents processing spikes by waiting until you stop dragging the window."""
        if self.raw_pil_image:
            if self.resize_after_id:
                self.after_cancel(self.resize_after_id)
            # Debounce: wait 150ms after the window stops resizing to calculate sharp details
            self.resize_after_id = self.after(150, lambda: self.update_preview_resolution(fast=False))

    def open_full_resolution(self):
        if self.selected_file_path is None:
            return

        pop = ctk.CTkToplevel(self)
        pop.title(f"Maximized High-Res - {os.path.basename(self.selected_file_path)}")
        pop.after(100, lambda: pop.focus())

        screen_w = pop.winfo_screenwidth()
        screen_h = pop.winfo_screenheight()

        pop_w = int(screen_w * 0.9)
        pop_h = int(screen_h * 0.9)
        pop.geometry(f"{pop_w}x{pop_h}")

        # For the big popup window, we reload the full high-res file using pristine LANCZOS mode
        full_hd_raw = Image.open(self.selected_file_path)
        full_hd_raw.thumbnail((pop_w - 60, pop_h - 60), Image.Resampling.LANCZOS)
        
        ctk_img_pop = ctk.CTkImage(light_image=full_hd_raw, dark_image=full_hd_raw, size=(full_hd_raw.width, full_hd_raw.height))
        
        pop_scroll = ctk.CTkScrollableFrame(pop)
        pop_scroll.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        pop_label = ctk.CTkLabel(pop_scroll, image=ctk_img_pop, text="")
        pop_label.pack(expand=True, pady=10)

    def clear_preview(self):
        self.raw_pil_image = None
        self.selected_file_path = None
        self.pop_btn.configure(state="disabled")
        self.preview_label.configure(image="", text="Select a PNG file to view it")

    def go_up(self):
        parent_dir = os.path.dirname(self.current_dir)
        if parent_dir != self.current_dir:
            self.load_directory(parent_dir)

    def go_to_entered_path(self, event):
        target_path = self.path_entry.get()
        if os.path.isdir(target_path):
            self.load_directory(target_path)
        else:
            messagebox.showerror("Invalid Path", "That directory does not exist.")

    def on_list_double_click(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return

        selected_item = self.listbox.get(selection[0])
        parts = selected_item.split("  ", 1)
        if len(parts) < 2:
            return
        clean_name = parts[1]

        target_path = os.path.join(self.current_dir, clean_name)
        if os.path.isdir(target_path):
            self.load_directory(target_path)
            
def open_viewer(target_folder="cards"):
    """Instantly opens the optimized modern card browser window."""
    app = FastModernCardViewer(target_folder=target_folder)
    app.mainloop()

if __name__ == "__main__":
    open_viewer()