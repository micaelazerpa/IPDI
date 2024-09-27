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

imgIO = None
imRGB = None
process =None
image = None
im= None

class Application(tk.Frame):
    global url_image
    url_image =""

    global imRGB

    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()
        self.loaded_image = None

    def create_widgets(self):
        global process
        # Frame que contendrá los dos cuadros y los botones
        self.squares_frame = tk.Frame(self)
        self.squares_frame.pack(side="top", fill="both", expand=True)

        # Cuadro 1 de 500x500 píxeles
        self.square1 = tk.Frame(self.squares_frame, width=500, height=400, bg="lightblue")
        self.square1.pack(side="left", padx=10, pady=10)
        
        # Cuadro 2 de 500x500 píxeles 
        self.square2 = tk.Canvas(self.squares_frame, width=500, height=400, bg="lightblue")
        self.square2.pack(side="left", padx=10, pady=10)

        # Crear un frame inferior
        self.operation = tk.Frame(self)
        self.operation.pack(side="bottom", fill="x", pady=15)
        # Frame para los botones inferiores
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side="bottom", fill="x")
        
        # Botón para salir
        self.out = tk.Button(self.bottom_frame, text="Salir", command=self.close, height=2, width=20)
        self.out.pack(side="right", padx=10)

        # Guardar
        self.save = tk.Button(self.bottom_frame, text="Guardar", height=2, width=20, command=self.save_image)
        self.save.pack(side="right", pady=5)

        # Procesar
        self.process = tk.Button(self.bottom_frame, command=self.process_arithmetic, text='Procesar', height=2, width=28)
        self.process.pack(side="right", padx=10)

        self.message= tk.Label(self.bottom_frame, text=f"Imágen subida en {process}")
        self.message.pack(side="top")

        # Botones para subir imagenes
        self.buttonA = tk.Button(self.bottom_frame, command=lambda: self.upload_image(), text='Subir Imagen', height=2, width=20)
        self.buttonA.pack(side="left", padx=10)


        # Crea el Combobox para Operaciones
        options = ["Pasabajos llano 3x3", "Pasabajos llano 5x5", "Pasabajos llano 7x7", "Bartlett 3x3", "Bartlett 5x5", "Bartlett 7x7"]
        self.operation_message= tk.Label(self.bottom_frame, text="Filtros")
        self.operation_message.pack(side="left")
        self.comboboxOperations = ttk.Combobox(self.bottom_frame, values=options, height=20, width=28)
        self.comboboxOperations.set("Pasabajos llano 3x3")  
        self.comboboxOperations.pack(side="left", padx=10)
        

    def upload_image(self):
        global url_image, process, image

        url_image = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        print(f"Upload de imagen seleccionada: {url_image}")
        
        if url_image: 
            process = 'RGB'
            self.message.config(text=f"Imágen guardada en {process}")
            self.show_image(url_image)
        image = url_image

    def show_image(self, url_image):
        global im, imgIO
        print(f"Show de imagen seleccionada: {url_image}")

        imgIO = imageio.imread(url_image)
        img=Image.fromarray(imgIO)
        new_img = img.resize((500, 400))  # Cambiado para que la imagen se ajuste al tamaño del cuadro
        imagen_tk = ImageTk.PhotoImage(new_img)
        
        img1 = Label(self.square1, image=imagen_tk)
        im=imgIO

        img1.image = imagen_tk
        img1.pack()

    def close(self):
        response = messagebox.askquestion("Salir", "¿Desea salir de la interfaz?")
        if response == 'yes':
            self.master.destroy()

    def save_image(self):
        if self.loaded_image: 
            save_path = os.path.join(os.getcwd(), 'imagen_guardada.png')  # Guarda en la carpeta de ejecución
            self.loaded_image.save(save_path)  
            messagebox.showinfo("Imagen guardada", f"Imagen guardada en {save_path}")
        else:
            messagebox.showwarning("Sin imagen", "No hay imagen para guardar.")

    def processImageRGB(self):
        global imgIO
        #1.Normalizar los valores de RGB del pixel
        im = np.clip(imgIO /255.,0.,1.)
        print(im.shape, im.dtype)

        titles = ['RGB','Rojo','Verde','Azul']
        chanels = ['','Reds','Greens','Blues']

        for i in range(4):
            plt.subplot(1,4,i+1)
            if i==0:
                plt.imshow(im)
            else:
                plt.imshow(im[:,:,i-1], cmap=chanels[i])
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

    def imageRGBtoYIQ(self, image):
        global process
        #   1.Normalizar los valores de RGB del pixel
        im = np.clip(image/255,0.,1.)
        
        #   2.RGB -> YIQ (utilizando la segunda matriz)
        YIQ=np.zeros(im.shape)

        YIQ[:, :, 0] = np.clip((im[:, :, 0] * 0.299 + im[:, :, 1] * 0.587 + im[:, :, 2] * 0.114), 0., 1.)
        YIQ[:, :, 1] = np.clip(im[:, :, 0] * 0.595 + im[:, :, 1] * (-0.274) + im[:, :, 2] * (-0.321), -0.59, 0.59)
        YIQ[:, :, 2] = np.clip(im[:, :, 0] * 0.211 + im[:, :, 1] * (-0.522) + im[:, :, 2] * (0.311), -0.52, 0.52) 
        
        process = 'YIQ'
        self.message.config(text=f"Imágen guardada en {process}")
        
        return YIQ
    
    def imageYIQtoRGB(self, YIQ):        
        #   7. Y’I’Q’ -> R’G’B’ (el RGB normalizado del pixel procesado)
        print('Entro a convertir en RGB')
        
        RGB=np.zeros(YIQ.shape)
        RGB[:,:,0] = np.clip(YIQ[:,:,0] + 0.9563 * YIQ[:,:,1] + 0.6210 * YIQ[:,:,2], 0, 1)
        RGB[:,:,1] = np.clip(YIQ[:,:,0] - 0.2721 * YIQ[:,:,1] - 0.6474 * YIQ[:,:,2], 0, 1)
        RGB[:,:,2] = np.clip(YIQ[:,:,0] - 1.1070 * YIQ[:,:,1] + 1.7046 * YIQ[:,:,2], 0, 1)

        return RGB

    def processImageRGBtoBYTE(self, image):
        # Convierte el tipo de dato a uint8 (enteros sin signo de 8 bits)
        if image is None or image.size == 0:
           messagebox.showinfo("Error", "No se ha cargado ninguna imagen.")
           return
        #   8.Convertir R’G’B’ a bytes y graficar el pixel
        RGB_BYTE = np.uint8(image * 255)
        return RGB_BYTE
    
    def K_lineal(self, dim):
        print(f"Lineal")
        K= np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                K[i,j]=1/(dim*dim)
        return K

    def bartlett(self, dim):
        global imRGB
        print(f"Bartlett")
        
        newArray = (dim+1)//2-np.abs(np.arange(dim)-dim//2)
        K = np.outer(newArray,newArray.T)
        K = K/K.sum()
        return K

    def convolucion(self, image, kernel=np.ones((1,1))):
        global imRGB
        print(f"Kernel", kernel)
        YIQ = self.imageRGBtoYIQ(image)
        image=YIQ[:,:,0]

        conv = np.zeros((np.array(image.shape) - np.array(kernel.shape) + 1))

        for i in range(conv.shape[0]):
            for j in range(conv.shape[1]):
                conv[i,j]= (image[i:i+kernel.shape[0], j:j+kernel.shape[1]]*kernel).sum()
        print(f"Redimensiones imagen-----: {conv.shape}")

        #RGB= self.imageYIQtoRGB(conv)
        imRGB = conv


    # Función para procesar la operación según la selección
    def process_arithmetic(self):
        global im, imRGB
        selection= self.comboboxOperations.get()
        print(f"Operación seleccionada: {selection}")
        if im is not None:
            #kernel = np.ones((3,3))
            #kernel /=np.sum(kernel)

            match selection:
                case "Pasabajos llano 3x3":
                    kernel = self.K_lineal(3)
                    self.convolucion(im, kernel)
                case "Pasabajos llano 5x5":
                    kernel = self.K_lineal(5)
                    self.convolucion(im, kernel)
                case "Pasabajos llano 7x7":
                    kernel = self.K_lineal(7)
                    self.convolucion(im, kernel)
                case "Bartlett 3x3":
                    kernel = self.bartlett(3)
                    self.convolucion(im, kernel)
                case "Bartlett 5x5":
                    kernel = self.bartlett(5)
                    self.convolucion(im, kernel)
                case "Bartlett 7x7":
                    kernel = self.bartlett(7)
                    self.convolucion(im, kernel)
                case _:
                    print("Opción inválida")

            # Convertir el array NumPy resultante a una imagen Pillow
            img = Image.fromarray(np.uint8(imRGB * 255))  
            new_img = img.resize((500, 400))  
            
            # Convertir la imagen a formato Tkinter
            imagen_tk = ImageTk.PhotoImage(new_img)
            
            # Mostrar la imagen en el square2
            img1 = Label(self.square2, image=imagen_tk)
            img1.image = imagen_tk  # Mantener una referencia de la imagen
            img1.pack()         
            self.loaded_image = img

root = tk.Tk()
root.geometry('1300x800')  # Ajuste del tamaño de la ventana para acomodar los cuadros y botones
app = Application(master=root)
app.mainloop()
