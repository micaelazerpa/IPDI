import tkinter as tk
from tkinter import filedialog, Label, messagebox 
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Frame para la imagen
        self.image_frame = tk.Frame(self)
        self.image_frame.pack(side="top", fill="both", expand=True)
        
        # Frame para los botones
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side="bottom", fill="x")

        # Botón para subir imagen
        self.button = tk.Button(self.bottom_frame, command=self.upload_image, text='Subir Imagen', height=2, width=28)
        self.button.pack(side="left", padx=10)

        # Botón para salir
        self.out = tk.Button(self.bottom_frame, text="Salir", command=self.close, height=2, width=28)
        self.out.pack(side="right", padx=10)
        #self.button.place(x=100, y=350, anchor="sw")

    def upload_image(self):
        self.url_image = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
        )
        if self.url_image: 
            #print(self.url_image)
            self.show_image(self.url_image)

    def show_image(self, url_image):

        img = Image.open(url_image)
        new_img = img.resize((300, 256))
        imagen_tk = ImageTk.PhotoImage(new_img)
        img1 = Label(self.image_frame, image=imagen_tk)
        img1.image = imagen_tk
        img1.place(x=10, y=38)
        img1.pack(side="top", pady=10)

    def close(self):
        response = messagebox.askquestion("Salir", "¿Desea salir de la intefaz?")
        if response == 'yes':
            self.master.destroy()
            


root = tk.Tk()
root.geometry('700x500')
app = Application(master=root)
app.mainloop()