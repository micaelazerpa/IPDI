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
image_show = None
image_show1 = None
image= None

class Application(tk.Frame):
    global url_image

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
        
        # Frame para los botones entre los cuadros
        self.button_centro = tk.Frame(self.squares_frame)
        self.button_centro.pack(side="left", padx=10, pady=10)
        
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
        
        # Copiar
        self.button3 = tk.Button(self.button_centro, command=self.copy_image,  text="<- Copiar", height=2, width=20)
        self.button3.pack(pady=20)

        # Procesar
        self.process = tk.Button(self.button_centro, command=self.process_arithmetic, text='Procesar', height=2, width=20)
        self.process.pack(side="right", padx=10)

        self.message= tk.Label(self.bottom_frame, text=f"Imágen subida en {process}")
        self.message.pack(side="top")

        # Botones para subir imagenes
        self.buttonA = tk.Button(self.bottom_frame, command=lambda: self.upload_image(), text='Subir Imagen', height=2, width=20)
        self.buttonA.pack(side="left", padx=10)

        self.bottom_histogram = tk.Button(self.bottom_frame, command=self.histogram, text='Histograma', height=2, width=15)
        self.bottom_histogram.pack(side="left", padx=10)        

        self.label_umbral = tk.Label(self.bottom_frame, text="Umbral")
        self.label_umbral.pack(side="left")
        self.input_umbral = tk.Entry(self.bottom_frame)
        self.input_umbral.pack(side="left", padx=10)

        # Crea el Combobox para Operaciones
        options = ["50% de pixeles negros y blancos","Distancia mínima", "Otsu", "Laplaciano", "Borde morfológico", "Marching squares"]
        self.operation_message= tk.Label(self.bottom_frame, text="Filtro")
        self.operation_message.pack(side="left")
        self.comboboxOperations = ttk.Combobox(self.bottom_frame, values=options, height=20, width=28)
        self.comboboxOperations.set("Otsu")  
        self.comboboxOperations.pack(side="left", padx=10)
        
    def upload_image(self):
        global url_image, process, image

        url_image = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        print(f"Upload de imagen seleccionada: {url_image}")
        
        if url_image: 
            process = 'RGB'
            self.message.config(text=f"Imágen guardada en {process}")
            self.show_image(url_image)

    def show_image(self, url_image):
        global image, image_show
        print(f"Show de imagen seleccionada: {url_image}")
         # Limpiar la imagen anterior si existe
        if image_show is not None:
            image_show.destroy()

        image = imageio.imread(url_image)
        img=Image.fromarray(image)
        new_img = img.resize((500, 400))  # Cambiado para que la imagen se ajuste al tamaño del cuadro
        imagen_tk = ImageTk.PhotoImage(new_img)
        
        image_show = Label(self.square1, image=imagen_tk)
        image_show.image = imagen_tk
        image_show.pack()

    def close(self):
        response = messagebox.askquestion("Salir", "¿Desea salir de la interfaz?")
        if response == 'yes':
            self.master.destroy()

    def save_image(self):
        global loaded_image
        if loaded_image is not None:             
            if loaded_image.mode == 'F':
                # Si es modo 'F', convertir a 'L' (escala de grises)
                loaded_image = loaded_image.convert('L')

            save_path = os.path.join(os.getcwd(), 'imagen_guardada.bmp')  # Guardar en la carpeta actual
            loaded_image.save(save_path)  
            messagebox.showinfo("Imagen guardada", f"Imagen guardada en {save_path}")
        else:
            messagebox.showwarning("Sin imagen", "No hay imagen para guardar.")

    def copy_image(self):
        global loaded_image, image_show
        if image_show is not None:
            image_show.destroy()
 
        if loaded_image is not None:
            # Redimensionar la imagen para el cuadro
            new_img = loaded_image.resize((500, 400))
            imagen_tk = ImageTk.PhotoImage(new_img)
            
            # Mostrar la imagen en square1
            image_show = Label(self.square1, image=imagen_tk)
            image_show.image = imagen_tk  # Mantener la referencia
            image_show.pack()
        else:
            print("No hay imagen para copiar")

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
        im = np.clip(image/255,0.,1.)

        if len(image.shape) == 2:  # Si es una imagen en escala de grises
            print("La imagen está en escala de grises, no se convierte a YIQ.")
            return image
        #   1.Normalizar los valores de RGB del pixel
        
        #   2.RGB -> YIQ (utilizando la segunda matriz)
        YIQ=np.zeros(im.shape)

        YIQ[:, :, 0] = np.clip((im[:, :, 0] * 0.299 + im[:, :, 1] * 0.587 + im[:, :, 2] * 0.114), 0., 1.)
        YIQ[:, :, 1] = np.clip(im[:, :, 0] * 0.595 + im[:, :, 1] * (-0.274) + im[:, :, 2] * (-0.321), -0.59, 0.59)
        YIQ[:, :, 2] = np.clip(im[:, :, 0] * 0.211 + im[:, :, 1] * (-0.522) + im[:, :, 2] * (0.311), -0.52, 0.52) 
        
        process = 'YIQ'
        self.message.config(text=f"Imágen guardada en {process}")
        
        return YIQ
    
    def imageYIQtoRGB(self,YIQop, YIQ):        
        #   7. Y’I’Q’ -> R’G’B’ (el RGB normalizado del pixel procesado)
        print('Entro a convertir en RGB')
        if YIQop is None:
            YIQop = YIQ[:,:,0]
        
        RGB=np.zeros(YIQ.shape)
        RGB[:,:,0] = np.clip(YIQop + 0.9563 * YIQ[:,:,1] + 0.6210 * YIQ[:,:,2], 0, 1)
        RGB[:,:,1] = np.clip(YIQop - 0.2721 * YIQ[:,:,1] - 0.6474 * YIQ[:,:,2], 0, 1)
        RGB[:,:,2] = np.clip(YIQop - 1.1070 * YIQ[:,:,1] + 1.7046 * YIQ[:,:,2], 0, 1)

        return RGB

    def processImageRGBtoBYTE(self, image):
        # Convierte el tipo de dato a uint8 (enteros sin signo de 8 bits)
        if image is None or image.size == 0:
           messagebox.showinfo("Error", "No se ha cargado ninguna imagen.")
           return
        #   8.Convertir R’G’B’ a bytes y graficar el pixel
        RGB_BYTE = np.uint8(image * 255)
        return RGB_BYTE

    def histogram(self):
        global image
        print(f"Histograma")
        if image.ndim == 3:
            YIQ = self.imageRGBtoYIQ(image)
            im = YIQ[:, :, 0]
        else:
            im = np.clip(image/255,0.,1.)
            
        histograma, bins = np.histogram(im.flatten(), bins=100, range=(0, 1))

        plt.subplots(figsize=(4, 2))
        plt.bar(bins[:-1], (histograma / histograma.sum()) * 100, width=(bins[1] - bins[0]), edgecolor='black')
        plt.title('Histograma')
        plt.xlabel('Luminancia')
        plt.ylabel('Frecuencia %')
        plt.show()

        return histograma
        
    def im_binaria(self, image):
        global imRGB
        im_bin = np.zeros(image.shape)

        print(f"Binaria")
        print(f"Rango de valores en la imagen original: {np.min(image)} a {np.max(image)}")
        umbral = float(self.input_umbral.get()) if self.input_umbral.get().strip() else 0.5
        print(f"Umbral", umbral)

        im_bin = np.where(image > umbral, 1, 0)
        #plt.imshow(im_bin, cmap='gray')
        #plt.show()
        imRGB = im_bin

    def otsu(self, img2, hist):
        global imRGB
        print(f"Otsu")
        print(f"Rango de valores en la imagen original: {np.min(img2)} a {np.max(img2)}")

        pixel_numb = img2.shape[0] * img2.shape[1]
        prom_pond = 1/pixel_numb
        bins = 100

        final_thresh = -1
        final_value = -1
        intensity = np.arange(100)
        for x in range (1,100):
            pcb = np.sum(hist[:x])
            pcf = np.sum(hist[x:])
            wb = pcb * prom_pond
            wf = pcf * prom_pond
            
            mub = np.sum(intensity[:x] * hist[:x]) / (pcb)
            muf = np.sum(intensity[x:] * hist[x:]) / (pcf)
            
            value = wb * wf * (mub-muf) **2
            if value > final_value:
                final_thresh = x/100
                final_value = value

        print(final_thresh)
        img8 = img2.copy()
        img8[img2 > final_thresh] = 1
        img8[img2 < final_thresh] = 0
        print(f"Resultado Otsu")
        
        plt.figure(6)
        plt.title('Binarizacion por el método de Otsu')
        plt.imshow(img8,'gray')
        plt.show()
        imRGB = img8

    def im_border_int(im, se):
        return im - im_erode(im, se)

    def im_gradient(im, se):
        return im_dilate(im,se) - im_erode(im,se)
    
    # Función para procesar la operación según la selección
    def process_arithmetic(self):
        global image, imRGB, image_show1, loaded_image
        selection= self.comboboxOperations.get()
        print(f"Operación seleccionada: {selection}")
        if image is not None:

            hist = self.histogram()
            if image.ndim == 3:
                YIQ = self.imageRGBtoYIQ(image)
                image = YIQ[:, :, 0] 
            else:
                image = np.clip(image / 255, 0., 1.) 

            match selection:
                case "50% de pixeles negros y blancos":
                    self.im_binaria(image)
                case "Distancia mínima":
                    kernel = self.box(5)
                    self.erosion(image, kernel)
                case "Otsu":
                    self.otsu(image, hist)
                case "Laplaciano":
                    kernel = self.circle(3)
                    self.dilatacion(image, kernel)
                case "Borde morfológico":
                    kernel = self.box(5)
                    self.dilatacion(image, kernel)
                case "Marching squares":
                    kernel = self.box(3)
                    self.mediana(image, kernel)
                case _:
                    print("Opción inválida")
            
            if image_show1 is not None:
                image_show1.destroy()
            # Convertir el array NumPy resultante a una imagen Pillow
            img = Image.fromarray(np.uint8(imRGB * 255))  
            new_img = img.resize((500, 400))  
            
            # Convertir la imagen a formato Tkinter
            imagen_tk = ImageTk.PhotoImage(new_img)
            
            # Mostrar la imagen en el square2
            image_show1 = Label(self.square2, image=imagen_tk)
            image_show1.image = imagen_tk  # Mantener una referencia de la imagen
            image_show1.pack()      
            loaded_image = img

root = tk.Tk()
root.geometry('1300x800')  # Ajuste del tamaño de la ventana para acomodar los cuadros y botones
app = Application(master=root)
app.mainloop()
