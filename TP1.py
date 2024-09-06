import tkinter as tk
from tkinter import filedialog, Label, messagebox
import random as R
from tkinter import *
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
from PIL import Image, ImageTk

import os

class Application(tk.Frame):
    global url_image
    url_image =""
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
        self.button1 = tk.Button(self.buttons_frame, text="Guardar", height=2, width=20, command=self.save_image)
        self.button1.pack(pady=5)

        # Botón 2
        self.button2 = tk.Button(self.buttons_frame, text="Procesar", height=2, width=20)
        self.button2.pack(pady=5)
        saturacion='0'
        saturaciones = ['10*10','20*20','50*50','100*100']
        self.select= ttk.Combobox(self, textvariable=saturacion,values=saturaciones,height=12,width=20)
        self.select.pack(pady=5)
        self.select.set(saturacion)
        self.seleccionado=tk.Label(self,textvariable=saturacion)
        self.seleccionado.pack(pady=5)

        # Botón 3
        self.button3 = tk.Button(self.buttons_frame, command=self.processImageRGB, text="Botón RGB", height=2, width=20)
        self.button3.pack(pady=5)

        # Botón 4
        self.button4 = tk.Button(self.buttons_frame, command=self.processImageYIQ, text="Botón YIQ", height=2, width=20)
        self.button4.pack(pady=5)

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

        # Crear 4 subcuadros dentro de square2 (cada uno de 250x250)
        self.square2.create_rectangle(5, 5, 250, 250, outline="blue", width=2)  # Cuadro superior izquierdo
        self.square2.create_rectangle(250, 5, 495, 250, outline="blue", width=2)  # Cuadro superior derecho
        self.square2.create_rectangle(5, 250, 250, 495, outline="blue", width=2)  # Cuadro inferior izquierdo
        self.square2.create_rectangle(250, 250, 495, 495, outline="blue", width=2)  # Cuadro inferior derecho
        self.square2_rect1 = (5, 5, 250, 250)     # Subcuadro 1 (superior izquierdo)
        self.square2_rect2 = (255, 5, 495, 250)   # Subcuadro 2 (superior derecho)
        self.square2_rect3 = (5, 255, 250, 495)   # Subcuadro 3 (inferior izquierdo)
        self.square2_rect4 = (255, 255, 495, 495) 

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
        global url_image
        url_image = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        print(f"Upload de imagen seleccionada: {url_image}")
        
        if url_image: 
            self.show_image(url_image)

    def show_image(self, url_image):
        print(f"Show de imagen seleccionada: {url_image}")

        imgIO = imageio.imread(url_image)
        img=Image.fromarray(imgIO)
        new_img = img.resize((500, 500))  # Cambiado para que la imagen se ajuste al tamaño del cuadro
        imagen_tk = ImageTk.PhotoImage(new_img)
        img1 = Label(self.square1, image=imagen_tk)
        img1.image = imagen_tk
        img1.pack()
        self.loaded_image = img 

    def close(self):
        response = messagebox.askquestion("Salir", "¿Desea salir de la interfaz?")
        if response == 'yes':
            self.master.destroy()

    def save_image(self):
        if self.loaded_image:
            save_path = os.path.join(os.getcwd(), 'imagen_guardada.png')  # Guarda en la carpeta de ejecución
            self.loaded_image.save(save_path)
            messagebox.showinfo("Imagen guardada", f"Imagen guardada en {save_path}")
            self.url_image = save_path  # Actualiza la ruta de la imagen guardada

    def processImageRGB(self):
        global url_image
        #if not self.url_image:
         #   messagebox.showwarning("Error", "No se ha cargado ninguna imagen.")
           # return
        print(f"Ruta de imagen seleccionada: {url_image}")
        #im = imageio.v2.imread(self.url_image.split('\\').reverse)
        im = imageio.imread(url_image)

        print(im.shape, im.dtype)

        titles = ['Rojo','Verde','Azul']
        chanels = ['Reds','Greens','Blues']

        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.imshow(im[:,:,i], cmap=chanels[i])
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

    def processImageYIQ(self):
        im = imageio.imread(url_image)
        im = np.clip(im/255,0.,1.)
        YIQ=np.zeros(im.shape)
        YIQ[:,:,0] = np.clip((im[:,:,0]*0.299 +  im[:,:,1] *0.587 + im[:,:,2]*0.114),0.,1.)
        print(im.shape,im.dtype)
        YIQ[:,:,1] = np.clip(im[:,:,0]*0.595 +  im[:,:,1] *(-0.274) + im[:,:,2]*(-0.321),-0.59,0.59)
        print(im.shape,im.dtype)
        YIQ[:,:,2] = np.clip(im[:,:,0]*0.211 +  im[:,:,1] *(-0.522) + im[:,:,2]*(0.311),-0.52,0.52)

        YIQ[:,:,1] = (YIQ[:,:,1] + 0.59) / 1.18
        YIQ[:,:,2] = (YIQ[:,:,2] + 0.52) / 1.04        
        plt.figure(0)
        plt.imshow(im)        
        plt.figure(1)
        plt.imshow(YIQ[:,:,0])
        plt.figure(2)
        plt.imshow(YIQ[:,:,1])
        plt.figure(3)
        plt.imshow(YIQ[:,:,2])
        plt.show()
        
    def show_image_in_rect(self, rect_coords, image_array):
        img = Image.fromarray(image_array)
        new_img = img.resize((245, 245))
        imagen_tk = ImageTk.PhotoImage(new_img)
        self.square2.create_image(
            (rect_coords[0] + rect_coords[2]) // 2,  # Coordenada x central
            (rect_coords[1] + rect_coords[3]) // 2,  # Coordenada y central
            image=imagen_tk
        )
        self.square2.image = imagen_tk
    
root = tk.Tk()
root.geometry('1300x800')  # Ajuste del tamaño de la ventana para acomodar los cuadros y botones
app = Application(master=root)
app.mainloop()
