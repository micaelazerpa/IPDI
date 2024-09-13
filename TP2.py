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
        self.button4 = tk.Button(self.buttons_frame, command=self.processImageYIQ, text="Botón RGB a YIQ", height=2, width=20)
        self.button4.pack(pady=5)

        # Botón 4
        # self.button6 = tk.Button(self.buttons_frame, command=self.imageYIQtoRGB, text="Botón YIQ a RGB", height=2, width=20)
        # self.button6.pack(pady=5)

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
            self.message.config(text=f"Imágenes guardadas en {process}")
            if (type == 'A'):
                self.show_image(url_image, type)
            else:
                self.show_image(url_image, type)
        image = url_image

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

    def imageRGBtoYIQ(self, image):
        #   1.Normalizar los valores de RGB del pixel
        im = np.clip(image/255,0.,1.)
        
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

    def processImageYIQ(self):
        global imA, imB, process

        # Procesar la primera imagen (imA)
        YIQ_A= self.imageRGBtoYIQ(imA)
        imA = self.processImageRGBtoBYTE(YIQ_A)

        # Procesar la segunda imagen (imB)
        YIQ_B= self.imageRGBtoYIQ(imB)
        imB = self.processImageRGBtoBYTE(YIQ_B)
        process='YIQ'
        self.message.config(text=f"Imagenes guardadas en {process}")
        
        titles = ['Canal YIQ', 'Canal Y', 'Canal I', 'Canal Q']

        plt.figure(figsize=(10, 5))
        # Visualización de la imagen A
        RGB_A= self.imageYIQtoRGB(YIQ_A)
        for i in range(4):
            plt.subplot(2, 4, i + 1)  # Fila 1 para imA
            if i == 0:
                plt.imshow(RGB_A)
            else:
                plt.imshow(RGB_A[:, :, i - 1])
            plt.title(f"Imagen A - {titles[i]}")
            plt.axis('off')

        # Visualización de la imagen B
        RGB_B= self.imageYIQtoRGB(YIQ_B)
        for i in range(4):
            plt.subplot(2, 4, i + 5)  # Fila 2 para imB
            if i == 0:
                plt.imshow(RGB_B)
            else:
                plt.imshow(RGB_B[:, :, i - 1])
            plt.title(f"Imagen B - {titles[i]}")
            plt.axis('off')

        plt.show()


    def resize_image(self, img, target_shape):
        img_resized = img.resize(target_shape, Image.Resampling.LANCZOS)
        return np.array(img_resized)
    
    def ensure_rgb(self, image):
        #Para que la imagen tenga 3 canales
        if image.shape[2] > 3:
            # Si hay más de 3 canales, solo toma los primeros 3
            image = image[:, :, :3]
        elif image.shape[2] < 3:
            # Si hay menos de 3 canales, convierte la imagen a RGB
            image = np.dstack([image] * 3)
        return image
    
    # Funciones para cada operación
    def suma_clampeada(self, A, B):
        global process
        print("Ejecutando Suma Clampeada")
        
        C = np.zeros(A.shape)

        if (process == 'RGB'):
            print("Ejecutando Suma de RGB")
            A = (A/255.0)
            B = (B/255.0)
            C = np.clip(A + B)

        if (process == 'YIQ'):
            print("Ejecutando Suma de YIQ")
            C [:,:,0]= np.clip((A[:, :, 0] + B[:, :, 0]),0.,1.)
            C [:,:,1] = ((A[:,:,0] * A[:,:,1]) + (B[:,:,0] * B[:,:,1])) / (A[:,:,0] + B[:,:,0])
            C [:,:,2] = ((A[:,:,0] * A[:,:,2]) + (B[:,:,0] * B[:,:,2])) / (A[:,:,0] + B[:,:,0])
            C= self.imageYIQtoRGB(C)

        plt.imshow(C)
        plt.title('Suma Clampeada')
        plt.show()

    def resta_clampeada(self, A, B):
        global process

        print("Ejecutando Resta Clampeada")
        A = (A/255.0)
        B = (B/255.0)
        C = np.zeros(A.shape)

        if (process == 'RGB'):
            print("Ejecutando Resta de RGB")
            C = np.clip(A - B, 0, 255)
        
        if (process == 'YIQ'):
            print("Ejecutando Resta de YIQ")
            C [:,:,0]= np.clip(A[:, :, 0] + B[:, :, 0], 0., 1.)
            C [:,:,1] = ((A[:,:,0] * A[:,:,1]) - (B[:,:,0] * B[:,:,1])) / (A[:,:,0] + B[:,:,0])
            C [:,:,2] = ((A[:,:,0] * A[:,:,2]) - (B[:,:,0] * B[:,:,2])) / (A[:,:,0] + B[:,:,0])
            C= self.imageYIQtoRGB(C)
    
        plt.imshow(C)
        plt.title('Resta Clampeada')
        plt.show()

    def suma_promediada(self, A, B):
        global process

        print("Ejecutando Suma Promediada")
        A = (A/255.0)
        B = (B/255.0)
        C = np.zeros(A.shape)

        if (process == 'RGB'):
            print("Ejecutando Suma de RGB")
            C = np.clip((A + B)/2)
        
        if (process == 'YIQ'):
            print("Ejecutando Suma de YIQ")
            C [:,:,0]= np.clip((A[:, :, 0] + B[:, :, 0])/2, 0., 1.)
            C [:,:,1] = ((A[:,:,0] * A[:,:,1]) + (B[:,:,0] * B[:,:,1])) / (A[:,:,0] + B[:,:,0]+ 1e-5)
            C [:,:,2] = ((A[:,:,0] * A[:,:,2]) + (B[:,:,0] * B[:,:,2])) / (A[:,:,0] + B[:,:,0]+ 1e-5)
            C= self.imageYIQtoRGB(C)
        
        plt.imshow(C)
        plt.title('Suma Promediada')
        plt.show()


    def resta_promediada(self, A, B):
        global process

        print("Ejecutando Resta Promediada")
        A = (A/255.0)
        B = (B/255.0)
        if (process == 'RGB'):
            print("Ejecutando Resta de RGB")
            C = np.clip((A - B)/2)
        
        if (process == 'YIQ'):
            print("Ejecutando Resta de YIQ")
            C [:,:,0]= np.clip((A[:, :, 0] + B[:, :, 0])/2, 0., 1.)
            C [:,:,1] = ((A[:,:,0] * A[:,:,1]) - (B[:,:,0] * B[:,:,1])) / (A[:,:,0] + B[:,:,0])
            C [:,:,2] = ((A[:,:,0] * A[:,:,2]) - (B[:,:,0] * B[:,:,2])) / (A[:,:,0] + B[:,:,0])
            C= self.imageYIQtoRGB(C)

        plt.imshow(C)
        plt.title('Resta Promediada')
        plt.show()


    def producto(self, A, B):
        print("Ejecutando Producto")
        C = np.zeros(A.shape)
        A_normalized = A / 255.0
        B_normalized = B / 255.0
        C = A_normalized * B_normalized
        C = np.clip(C * 255, 0, 255).astype(np.uint8)
        plt.imshow(C)
        plt.title('Producto')
        plt.show()

    def cociente(self, A, B):
        print("Ejecutando Cociente")
        C = np.zeros(A.shape)
        A_normalized = A / 255.0
        B_normalized = B / 255.0
        C = A_normalized / B_normalized
        C = np.clip(C * 255, 0, 255).astype(np.uint8)
        plt.imshow(C)
        plt.title('Cociente')
        plt.show()
    def resta_valor_absoluto(self, A, B):
        print("Ejecutando Resta en Valor Absoluto")

    def if_darker(self, A, B):
        print("Ejecutando If Darker")
        YIQ_A = self.imageRGBtoYIQ(A)
        #YIQ_A = self.processImageRGBtoBYTE(YIQ_A)
        YIQ_B = self.imageRGBtoYIQ(B)
        #YIQ_B = self.processImageRGBtoBYTE(YIQ_B)
        
        YIQ_C = np.zeros(imA.shape)
        YIQ_C[:, :, 0] = np.where(YIQ_A[:, :, 0] < YIQ_B[:, :, 0], YIQ_A[:, :, 0], YIQ_B[:, :, 0])
        YIQ_C[:, :, 1] = np.where(YIQ_A[:, :, 0] < YIQ_B[:, :, 0], YIQ_A[:, :, 1], YIQ_B[:, :, 1])
        YIQ_C[:, :, 2] = np.where(YIQ_A[:, :, 0] < YIQ_B[:, :, 0], YIQ_A[:, :, 2], YIQ_B[:, :, 2])

        C= self.imageYIQtoRGB(YIQ_C)

        plt.imshow(C)
        plt.title('If Darker')
        plt.show()

    def if_lighter(self, A, B):
        print("Ejecutando If Lighter")
        YIQ_A = self.imageRGBtoYIQ(A)
        YIQ_B = self.imageRGBtoYIQ(B)
        
        YIQ_C = np.zeros(imA.shape)
        YIQ_C[:, :, 0] = np.where(YIQ_A[:, :, 0] > YIQ_B[:, :, 0], YIQ_A[:, :, 0], YIQ_B[:, :, 0])
        YIQ_C[:, :, 1] = np.where(YIQ_A[:, :, 0] > YIQ_B[:, :, 0], YIQ_A[:, :, 1], YIQ_B[:, :, 1])
        YIQ_C[:, :, 2] = np.where(YIQ_A[:, :, 0] > YIQ_B[:, :, 0], YIQ_A[:, :, 2], YIQ_B[:, :, 2])

        C= self.imageYIQtoRGB(YIQ_C)

        plt.imshow(C)
        plt.title('If Lighter')
        plt.show()
    # Función para procesar la operación según la selección
    def process_arithmetic(self):
        global imA, imB
        selection= self.comboboxOperations.get()
        print(f"Operación seleccionada: {selection}")

        if imA is not None and imB is not None:
            if imA.shape != imB.shape:
                # Redimensionar imágenes para que tengan el mismo tamaño
                new_size = (min(imA.shape[1], imB.shape[1]), min(imA.shape[0], imB.shape[0]))
                imA = self.resize_image(Image.fromarray(imA), new_size)
                imB = self.resize_image(Image.fromarray(imB), new_size)

            imA = self.ensure_rgb(imA)
            imB = self.ensure_rgb(imB)
                
            print(f"Redimensiones imagen A-----: {imA.shape}")
            print(f"Redimensiones imagen B-----: {imB.shape}")

            match selection:
                case "Suma clampeada":
                    self.suma_clampeada(imA, imB)
                case "Resta clampeada":
                    self.resta_clampeada(imA, imB)
                case "Suma promediada":
                    self.suma_promediada(imA, imB)
                case "Resta promediada":
                    self.resta_promediada(imA, imB)
                case "Producto":
                    self.producto(imA, imB)
                case "Cociente":
                    self.cociente(imA, imB)
                case "Resta en valor absoluto":
                    self.resta_valor_absoluto()
                case "If darker":
                    self.if_darker(imA, imB)
                case "If ligther":
                    self.if_lighter(imA, imB)
                case _:
                    print("Opción inválida")


root = tk.Tk()
root.geometry('1300x800')  # Ajuste del tamaño de la ventana para acomodar los cuadros y botones
app = Application(master=root)
app.mainloop()
