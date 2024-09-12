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

imA= None
imB= None

class Application(tk.Frame):
    global url_image
    url_image =""

    global imRGB

    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        global process
        # Frame que contendrá los dos cuadros y los botones
        self.squares_frame = tk.Frame(self)
        self.squares_frame.pack(side="top", fill="both", expand=True)

        # Cuadro 1 de 500x500 píxeles
        self.square1 = tk.Frame(self.squares_frame, width=500, height=500, bg="lightblue")
        self.square1.pack(side="left", padx=10, pady=10)

        # Cuadro 2 de 500x500 píxeles 
        self.square2 = tk.Canvas(self.squares_frame, width=500, height=500, bg="lightblue")
        self.square2.pack(side="left", padx=10, pady=10)

        # Frame para los botones entre los cuadros
        self.buttons_frame = tk.Frame(self.squares_frame)
        self.buttons_frame.pack(side="left", padx=10, pady=10)
        
   
        # Frame para los botones inferiores
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side="bottom", fill="x")

        # Botones para subir imagenes
        self.buttonA = tk.Button(self.bottom_frame, command=lambda: self.upload_image('A'), text='Subir Imagen A', height=2, width=28)
        self.buttonA.pack(side="left", padx=10)

        self.buttonB = tk.Button(self.bottom_frame, command=lambda: self.upload_image('B'), text='Subir Imagen B', height=2, width=28)
        self.buttonB.pack(side="left", padx=10)

        # Botón para salir
        self.out = tk.Button(self.bottom_frame, text="Salir", command=self.close, height=2, width=20)
        self.out.pack(side="right", padx=10)

        self.message= tk.Label(self.bottom_frame, text=f"Imagenes subidas en {process}")
        self.message.pack(side="top")

        # Botón 3
        self.button3 = tk.Button(self.buttons_frame, command=self.processImageRGB, text="Botón RGB", height=2, width=20)
        self.button3.pack(pady=5)

        # Inputs para los valores numéricos
        self.label_luminancia = tk.Label(self.buttons_frame, text="Luminancia")
        self.label_luminancia.pack(pady=5)
        self.input_luminancia = tk.Entry(self.buttons_frame)
        self.input_luminancia.pack(pady=5)

        self.label_saturacion = tk.Label(self.buttons_frame, text="Saturación")
        self.label_saturacion.pack(pady=5)
        self.input_saturacion = tk.Entry(self.buttons_frame)
        self.input_saturacion.pack(pady=5)

        # Botón 4
        self.button4 = tk.Button(self.buttons_frame, command=self.processImageYIQ, text="Botón YIQ", height=2, width=20)
        self.button4.pack(pady=5)

        # Botón 5
        self.button5 = tk.Button(self.buttons_frame, command=self.processImageRGBtoBYTE, text="Botón RGB a bytes", height=2, width=20)
        self.button5.pack(pady=5)

        # Botón 1
        self.button1 = tk.Button(self.buttons_frame, text="Guardar", height=2, width=20, command=self.save_image)
        self.button1.pack(pady=5)

        # Crea el Combobox para Operaciones
        options = ["Suma clampeada", "Resta clampeada", "Suma promediada", "Resta promediada", "Producto", "Cociente", "Resta en valor absoluto", "If darker", "If ligther"]
        self.comboboxOperations = ttk.Combobox(self.bottom_frame, values=options, height=20, width=28)
        self.comboboxOperations.set("Suma clampeada")  
        self.comboboxOperations.pack(side="left", padx=10)

        self.buttonC = tk.Button(self.bottom_frame, command=self.process_arithmetic, text='Procesar', height=2, width=28)
        self.buttonC.pack(side="left", padx=10)
        

    def upload_image(self, type):
        global url_image, process, image

        url_image = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        print(f"Upload de imagen seleccionada: {url_image}")
        
        if url_image: 
            process = 'RGB'
            self.message.config(text=f"Imagen guardada en {process}")
            if (type == 'A'):
                self.show_image(url_image, type)
            else:
                self.show_image(url_image, type)
        image = url_image
    
    def resize_image(self, img, target_shape):
        img_resized = img.resize(target_shape, Image.Resampling.LANCZOS)
        return np.array(img_resized)

    def show_image(self, url_image, type):
        global imA, imB, imgIO
        print(f"Show de imagen seleccionada: {url_image}")

        imgIO = imageio.imread(url_image)
        img=Image.fromarray(imgIO)
        new_img = img.resize((500, 500))  # Cambiado para que la imagen se ajuste al tamaño del cuadro
        imagen_tk = ImageTk.PhotoImage(new_img)

        if (type == 'A'):
            img1 = Label(self.square1, image=imagen_tk)
            imA=imgIO
        else:
            img1 = Label(self.square2, image=imagen_tk)
            imB=imgIO

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

    def processImageYIQ(self):
        global imRGB, process, imA, imB, imgIO
        #   1.Normalizar los valores de RGB del pixel
        im = np.clip(imgIO/255,0.,1.)
        
        #   2.RGB -> YIQ (utilizando la segunda matriz)
        YIQ=np.zeros(im.shape)

        YIQ[:, :, 0] = np.clip((im[:, :, 0] * 0.299 + im[:, :, 1] * 0.587 + im[:, :, 2] * 0.114), 0., 1.)
        YIQ[:, :, 1] = np.clip(im[:, :, 0] * 0.595 + im[:, :, 1] * (-0.274) + im[:, :, 2] * (-0.321), -0.59, 0.59)
        YIQ[:, :, 2] = np.clip(im[:, :, 0] * 0.211 + im[:, :, 1] * (-0.522) + im[:, :, 2] * (0.311), -0.52, 0.52) 

        a = float(self.input_luminancia.get()) if self.input_luminancia.get().strip() else 1
        b = float(self.input_saturacion.get()) if self.input_saturacion.get().strip() else 1

        #   3-5. Y’ := aY ;  con Y’ <= 1 (para que no se vaya de rango)
        #   4-6. I’ := bI ; con -0.5957 < I’ < 0.5957   Q’ := bQ ; con -0.5226 < Q’ < 0.5226

        YIQ[:,:,0] = np.clip(a * YIQ[:, :, 0], 0, 1)
        YIQ[:,:,1] = np.clip(b * YIQ[:,:,1], -0.5957, 0.5957)
        YIQ[:,:,2] = np.clip(b * YIQ[:,:,2], -0.5226, 0.5226)

        #   7. Y’I’Q’ -> R’G’B’ (el RGB normalizado del pixel procesado)
        img = np.clip(im /255.,0.,1.) 
        RGB=np.zeros(img.shape)
        RGB[:,:,0] = np.clip(YIQ[:,:,0] + 0.9563 * YIQ[:,:,1] + 0.6210 * YIQ[:,:,2], 0, 1)
        RGB[:,:,1] = np.clip(YIQ[:,:,0] - 0.2721 * YIQ[:,:,1] - 0.6474 * YIQ[:,:,2], 0, 1)
        RGB[:,:,2] = np.clip(YIQ[:,:,0] - 1.1070 * YIQ[:,:,1] + 1.7046 * YIQ[:,:,2], 0, 1)

        imRGB = RGB
        titles = ['Canal YIQ', 'Canal Y', 'Canal I', 'Canal Q']
        for i in range(4):
            plt.subplot(1,4,i+1)
            if i==0:
                plt.imshow(RGB)
                process='YIQ'
                self.message.config(text=f"Imagen guardada en {process}")
                imB = RGB
            else:
                plt.imshow(RGB[:,:,i-1])
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

    def processImageRGBtoBYTE(self):
        global imRGB
        if imRGB is None or imRGB.size() == 0:
           messagebox.showinfo("Error", "No se ha cargado ninguna imagen.")
           return
        #   8.Convertir R’G’B’ a bytes y graficar el pixel
        rgb_bytes = np.uint8(imRGB * 255)

        titles = ['Canal RGB procesado', 'Canal R', 'Canal G', 'Canal B']
        for i in range(4):
            plt.subplot(1,4,i+1)
            if i==0:
                plt.imshow(rgb_bytes)
            else:
                plt.imshow(rgb_bytes[:,:,i-1])
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

    # Funciones para cada operación
    def suma_clampeada(self):
        global imA, imB
        print("Ejecutando Suma Clampeada")
        
        imC = np.zeros(imA.shape)

        imC[:, :, 0] = np.clip(imA[:, :, 0] + imB[:, :, 0], 0., 1.)
        imC[:, :, 1] = np.clip(imA[:, :, 1] + imB[:, :, 1], 0., 1.)
        imC[:, :, 2] = np.clip(imA[:, :, 2] + imB[:, :, 2], 0., 1.)
        plt.imshow(imC)
        plt.show()

    def resta_clampeada(self):
        print("Ejecutando Resta Clampeada")

    def suma_promediada(self):
        print("Ejecutando Suma Promediada")

    def resta_promediada(self):
        print("Ejecutando Resta Promediada")

    def producto(self):
        print("Ejecutando Producto")

    def cociente(self):
        print("Ejecutando Cociente")

    def resta_valor_absoluto(self):
        print("Ejecutando Resta en Valor Absoluto")

    def if_darker(self):
        print("Ejecutando If Darker")

    def if_lighter(self):
        print("Ejecutando If Lighter")

    # Función para procesar la operación según la selección
    def process_arithmetic(self):
        global imA, imB
        selection= self.comboboxOperations.get()
        print(f"Operación seleccionada: {selection}")

        if imA is not None and imB is not None:
        # Asegúrate de que las imágenes tienen el mismo tamaño
            if imA.shape != imB.shape:
                # Redimensionar imágenes para que tengan el mismo tamaño
                new_size = (min(imA.shape[1], imB.shape[1]), min(imA.shape[0], imB.shape[0]))
                imA = self.resize_image(Image.fromarray(imA), new_size)
                imB = self.resize_image(Image.fromarray(imB), new_size)

            match selection:
                case "Suma clampeada":
                    self.suma_clampeada()
                case "Resta clampeada":
                    self.resta_clampeada()
                case "Suma promediada":
                    self.suma_promediada()
                case "Resta promediada":
                    self.resta_promediada()
                case "Producto":
                    self.producto()
                case "Cociente":
                    self.cociente()
                case "Resta en valor absoluto":
                    self.resta_valor_absoluto()
                case "If darker":
                    self.if_darker()
                case "If lighter":
                    self.if_lighter()
                case _:
                    print("Opción inválida")


root = tk.Tk()
root.geometry('1300x800')  # Ajuste del tamaño de la ventana para acomodar los cuadros y botones
app = Application(master=root)
app.mainloop()
