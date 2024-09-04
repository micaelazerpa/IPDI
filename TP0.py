import tkinter as tk
from tkinter import filedialog, Label, messagebox 
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Frame que contendrá los dos cuadros y los botones
        self.squares_frame = tk.Frame(self)
        self.squares_frame.pack(side="top", fill="both", expand=True)

        # Cuadro 1 de 500x500 píxeles
        self.square1 = tk.Frame(self.squares_frame, width=500, height=500, bg="lightblue")
        self.square1.pack(side="left", padx=10, pady=10)

        # Frame para los botones entre los cuadros
        self.buttons_frame = tk.Frame(self.squares_frame)
        self.buttons_frame.pack(side="left", padx=10, pady=10)

        # Botón 1
        self.button1 = tk.Button(self.buttons_frame, text="Botón 1", height=2, width=20)
        self.button1.pack(pady=5)

        # Botón 2
        self.button2 = tk.Button(self.buttons_frame, text="Botón 2", height=2, width=20)
        self.button2.pack(pady=5)

        # Botón 3
        self.button3 = tk.Button(self.buttons_frame, text="Botón 3", height=2, width=20)
        self.button3.pack(pady=5)

        # Cuadro 2 de 500x500 píxeles con borde de líneas de trazos
        self.square2 = tk.Canvas(self.squares_frame, width=500, height=500, bg="white")
        self.square2.pack(side="left", padx=10, pady=10)
        
        # Dibujar el borde con líneas de trazos
        self.square2.create_rectangle(
            5, 5, 495, 495,
            outline="gray",
            width=2,
            dash=(5, 2)
        )

        # Frame para los botones inferiores
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side="bottom", fill="x")

        # Botón para subir imagen
        self.button = tk.Button(self.bottom_frame, command=self.upload_image, text='Subir Imagen', height=2, width=28)
        self.button.pack(side="left", padx=10)

        # Botón para salir
        self.out = tk.Button(self.bottom_frame, text="Salir", command=self.close, height=2, width=28)
        self.out.pack(side="right", padx=10)

    def upload_image(self):
        self.url_image = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
        )
        if self.url_image: 
            self.show_image(self.url_image)

    def show_image(self, url_image):
        img = Image.open(url_image)
        new_img = img.resize((500, 500))  # Cambiado para que la imagen se ajuste al tamaño del cuadro
        imagen_tk = ImageTk.PhotoImage(new_img)
        img1 = Label(self.square1, image=imagen_tk)
        img1.image = imagen_tk
        img1.pack()

    def close(self):
        response = messagebox.askquestion("Salir", "¿Desea salir de la interfaz?")
        if response == 'yes':
            self.master.destroy()

root = tk.Tk()
root.geometry('1300x800')  # Ajuste del tamaño de la ventana para acomodar los cuadros y botones
app = Application(master=root)
app.mainloop()
