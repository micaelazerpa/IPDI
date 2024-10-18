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
im= None

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
        options = ["Binarizar","Erosión 3x3", "Erosión 5x5", "Dilatación 3x3", "Dilatación 5x5", "Mediana 3x3", "Mediana 5x5", "Apertura 3x3", "Apertura 5x5", "Cierre 3x3", "Cierre 5x5"]
        self.operation_message= tk.Label(self.bottom_frame, text="Filtro")
        self.operation_message.pack(side="left")
        self.comboboxOperations = ttk.Combobox(self.bottom_frame, values=options, height=20, width=28)
        self.comboboxOperations.set("Erosión 3x3")  
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
        global im, image_show
        print(f"Show de imagen seleccionada: {url_image}")
         # Limpiar la imagen anterior si existe
        if image_show is not None:
            image_show.destroy()

        im = imageio.imread(url_image)
        img=Image.fromarray(im)
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
        global im
        print(f"Histograma")
        YIQ = self.imageRGBtoYIQ(im)
        histograma, bins = np.histogram(YIQ[:,:,0].flatten(), bins=10, range=(0, 1))

        plt.subplots(figsize=(4, 2))
        plt.bar(bins[:-1], (histograma / histograma.sum()) * 100, width=(bins[1] - bins[0]), edgecolor='black')
        plt.title('Histograma')
        plt.xlabel('Luminancia')
        plt.ylabel('Frecuencia %')
        plt.show()
    
    def gaussiano(self, dim):
        def pascal_triangle(steps, last_layer = np.array([1])):
            if steps==1:
                return last_layer
        
            next_layer = np.array([1,*(last_layer[:-1]+last_layer[1:]),1])
            
            return pascal_triangle(steps-1,next_layer)
        
        a = pascal_triangle(dim)
        k = np.outer(a, a.T)
        return k / k.sum()
    
    def convolucion(self, image, kernel=np.ones((1,1))):
        global imRGB
        print(f"Kernel", kernel)
        YIQ = self.imageRGBtoYIQ(image)
        
        if YIQ.ndim == 3:
            image = YIQ[:, :, 0]
        else:
            image = YIQ

        conv = np.zeros((np.array(image.shape) - np.array(kernel.shape) + 1))

        for i in range(conv.shape[0]):
            for j in range(conv.shape[1]):
                conv[i,j]= (image[i:i+kernel.shape[0], j:j+kernel.shape[1]]*kernel).sum()
        print(f"Redimensiones imagen-----: {conv.shape}")

        #RGB= self.imageYIQtoRGB(conv)
        return conv
        
    def im_binaria(self, image):
        im_bin = np.zeros(image.shape)
        YIQ = self.imageRGBtoYIQ(image)
        
        if YIQ.ndim == 3:
            im = YIQ[:, :, 0]
        else:
            im = YIQ

        print(f"Binaria")
        print(f"Rango de valores en la imagen original: {np.min(im)} a {np.max(im)}")
        umbral = float(self.input_umbral.get()) if self.input_umbral.get().strip() else 0.5
        print(f"Umbral", umbral)

        im_bin = np.where(im > umbral, 1, 0)
        im = np.uint8(im_bin * 255)
        #plt.imshow(im_bin, cmap='gray')
        #plt.show()
        return im
    
    def erosion(self, image, kernel, size):
        convolucion_resultado = self.convolucion(image, kernel)
        erosion_resultado = np.zeros(convolucion_resultado.shape)

        for i in range(convolucion_resultado.shape[0]):
            for j in range(convolucion_resultado.shape[1]):
                # En lugar de realizar una suma de productos, tomamos el valor mínimo
                erosion_resultado[i, j] = np.min(convolucion_resultado[i:i + size, j:j + size])
        print(f"Erosion resultado-----: {erosion_resultado}")
        plt.imshow(erosion_resultado, "gray")
        plt.show()
        return erosion_resultado

    def dilatacion(self, image, size):
        # Implementa la lógica para la operación de dilatación con un kernel de tamaño 'size'
        pass

    def mediana(self, image, size):
        # Implementa la lógica para la operación de filtro mediana con una ventana de tamaño 'size'
        pass

    def apertura(self, image, size):
        # Implementa la lógica para la operación de apertura (erosión seguida de dilatación) con un kernel de tamaño 'size'
        pass

    def cierre(self, image, size):
        # Implementa la lógica para la operación de cierre (dilatación seguida de erosión) con un kernel de tamaño 'size'
        pass

    # Función para procesar la operación según la selección
    def process_arithmetic(self):
        global im, imRGB, image_show1, loaded_image
        selection= self.comboboxOperations.get()
        print(f"Operación seleccionada: {selection}")
        if im is not None:
            #im = np.clip(im/255,0.,1.)            
            #convolve = signal.convolve
            #conv = convolve(im, kernel, 'valid')
            #YIQ= self.imageRGBtoYIQ(im)
            #im=YIQ[:, :, 0]
            match selection:
                case "Binarizar":
                    im=self.im_binaria(im)
                case "Erosión 3x3":
                    kernel = self.gaussiano(3)
                    im = self.erosion(im, kernel, size=3)
                case "Erosión 5x5":
                    self.erosion(im, size=5)
                case "Dilatación 3x3":
                    self.dilatacion(im, size=3)
                case "Dilatación 5x5":
                    self.dilatacion(im, size=5)
                case "Mediana 3x3":
                    self.mediana(im, size=3)
                case "Mediana 5x5":
                    self.mediana(im, size=5)
                case "Apertura 3x3":
                    self.apertura(im, size=3)
                case "Apertura 5x5":
                    self.apertura(im, size=5)
                case "Cierre 3x3":
                    self.cierre(im, size=3)
                case "Cierre 5x5":
                    self.cierre(im, size=5)
                case _:
                    print("Opción inválida")
            #YIQ= self.imageRGBtoYIQ(im)
            #im=YIQ[:, :, 0]
            if image_show1 is not None:
                image_show1.destroy()
            # Convertir el array NumPy resultante a una imagen Pillow
            img = Image.fromarray(im)  
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
