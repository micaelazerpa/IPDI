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
        YIQ = self.imageRGBtoYIQ(image)
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
    
    def box(self, r):
        se = np.ones((r*2+1,r*2+1),dtype=np.bool)
        plt.title("Caja")
        plt.imshow(se)
        plt.show()
        return se

    def circle(self, r, threshold = 0.3):
        vec = np.linspace(-r, r, r*2+1)
        [x,y] = np.meshgrid(vec,vec)
        se = (x**2 + y**2)**0.5 < (r + threshold)
        plt.title("Círculo")
        plt.imshow(se)
        plt.show()
        return se
    
    def convolucion(self, image, kernel=np.ones((1,1))):
        global imRGB
        print(f"Kernel", kernel)
        if image.ndim == 3:
            YIQ = self.imageRGBtoYIQ(image)
            im = YIQ[:, :, 0]
        else:
            im = np.clip(image/255,0.,1.)

        conv = np.zeros((np.array(im.shape) - np.array(kernel.shape) + 1))

        for i in range(conv.shape[0]):
            for j in range(conv.shape[1]):
                conv[i,j]= (im[i:i+kernel.shape[0], j:j+kernel.shape[1]]*kernel).sum()
        print(f"Redimensiones imagen-----: {conv.shape}")

        #RGB= self.imageYIQtoRGB(conv)
        return conv
        
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
    
    def erosion(self, im, kernel):
        global imRGB

        # Creamos la imagen resultante de tamaño reducido debido a la convolución
        erosion_im = np.zeros((np.array(im.shape) - np.array(kernel.shape) + 1))

        # Aplicamos la erosión tomando el valor mínimo dentro de cada vecindad
        for i in range(erosion_im.shape[0]):
            for j in range(erosion_im.shape[1]):
                # Tomamos el valor mínimo en la vecindad definida por el kernel
                erosion_im[i, j] = np.min(im[i:i+kernel.shape[0], j:j+kernel.shape[1]])

        print(f"Redimensiones imagen erosionada: {erosion_im.shape}")        
        plt.imshow(erosion_im, "gray")
        plt.show()
        imRGB= erosion_im

    def dilatacion(self, im, kernel):
        global imRGB

        dilatacion_im = np.zeros((np.array(im.shape) - np.array(kernel.shape) + 1))

        # Aplicamos la dilatación tomando el valor máximo dentro de cada vecindad
        for i in range(dilatacion_im.shape[0]):
            for j in range(dilatacion_im.shape[1]):
                # Tomamos el valor máximo en la vecindad definida por el kernel
                dilatacion_im[i, j] = np.max(im[i:i+kernel.shape[0], j:j+kernel.shape[1]])

        print(f"Redimensiones imagen dilatada: {dilatacion_im.shape}")
        plt.imshow(dilatacion_im, "gray")
        plt.show()
        imRGB = dilatacion_im

    def mediana(self, image, kernel):
        global imRGB

        mediana_im = np.zeros((np.array(image.shape) - np.array(kernel.shape) + 1))

        for i in range(mediana_im.shape[0]):
            for j in range(mediana_im.shape[1]):
                vecindad = image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                mediana_im[i, j] = np.median(vecindad)

        print(f"Redimensiones imagen con filtro de mediana: {mediana_im.shape}")

        plt.imshow(mediana_im, "gray")
        plt.show()
        imRGB = mediana_im

    def apertura(self, image, kernel):
        global imRGB
        im_erosionada = self.erosion(image, kernel)
        apertura = self.dilatacion(im_erosionada, kernel)

        imRGB = apertura

    def cierre(self, image, kernel):
        global imRGB
        im_dilatada = self.dilatacion(image, kernel)
        cierre = self.erosion(im_dilatada, kernel)

        imRGB = cierre

    def _morph_gray(im, se, op):
        result = np.zeros(im.shape)
        offset = (np.array(se.shape)-1)//2
        im = np.pad(im,[(offset[0],offset[0]),(offset[1],offset[1])],'edge')
        for y, x in np.ndindex(result.shape):
            pixels = im[y: y + se.shape[0], x: x + se.shape[1]][se]
            result[y, x] = op(pixels)

        plt.imshow(result, "gray")
        plt.show()
        return result
    
    def im_border_ext(im, se):
        return im_dilate(im, se) - im

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
            if image.ndim == 3:
                YIQ = self.imageRGBtoYIQ(image)
                image = YIQ[:, :, 0] 
            else:
                image = np.clip(image / 255, 0., 1.) 

            match selection:
                case "Binarizar":
                    self.im_binaria(image)
                case "Erosión 3x3":
                    kernel = self.box(5)
                    self.erosion(image, kernel)
                case "Erosión 5x5":
                    kernel = self.circle(2)
                    self.erosion(image, kernel)
                case "Dilatación 3x3":
                    kernel = self.circle(3)
                    self.dilatacion(image, kernel)
                case "Dilatación 5x5":
                    kernel = self.box(5)
                    self.dilatacion(image, kernel)
                case "Mediana 3x3":
                    kernel = self.box(3)
                    self.mediana(image, kernel)
                case "Mediana 5x5":
                    kernel = self.circle(5)
                    self.mediana(image, kernel)
                case "Apertura 3x3":
                    kernel = self.circle(3)
                    #self._morph_gray(image, kernel, np.max)
                    self.apertura(image, kernel)
                case "Apertura 5x5":
                    kernel = self.circle(5)
                    self.apertura(image, kernel)
                case "Cierre 3x3":
                    kernel = self.circle(3)
                    self.cierre(image, kernel)
                case "Cierre 5x5":                    
                    kernel = self.circle(5)
                    self.cierre(image, kernel)
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
