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
        options = ["Pasabajos llano 3x3", "Pasabajos llano 5x5", "Pasabajos llano 7x7", "Bartlett 3x3", "Bartlett 5x5", "Bartlett 7x7", "Gaussiano 5x5", "Gaussiano 7x7", "Pasaaltos Laplaciano v4", "Pasaaltos Laplaciano v8", "Pasabanda Dog 5x5", "Sobel O", "Sobel N", "Sobel E", "Sobel S", "Sobel NO", "Sobel NE", "Sobel SO", "Sobel SE"]
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

    def gaussiano(self, dim):
        def pascal_triangle(steps, last_layer = np.array([1])):
            if steps==1:
                return last_layer
        
            next_layer = np.array([1,*(last_layer[:-1]+last_layer[1:]),1])
            
            return pascal_triangle(steps-1,next_layer)
        
        a = pascal_triangle(dim)
        k = np.outer(a, a.T)
        return k / k.sum()
    
    # Función para crear un kernel gaussiano
    def gauss(self,size, sigma):
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return g / g.sum()

    # Función para crear el kernel DoG
    def dog(self, size, fs=0.2, cs=0.4):
        return self.gauss(size, fs) - self.gauss(size, cs)
# from scipy import signal
# # use scipy version from now on
# convolve = signal.convolve
# plt.imshow(convolve(im,bartlett(3), 'valid'),'gray')
# plt.title('Bartlett')
# plt.show()

    def gaussian_blur_filter(image, size):
        pascal_row = []
        kernel = np.zeros((size, size))
        n = size - 1

        for k in range(size):
            coefficient = math.comb(n, k)
            pascal_row.append(coefficient)
            kernel[0, :] = pascal_row
            kernel[size - 1, :] = pascal_row
            kernel[:, 0] = pascal_row
            kernel[:, size - 1] = pascal_row

        for i in range(1, size - 1):
            for j in range(1, size - 1):
                kernel[i, j] = kernel[0, j] * kernel[i, 0]
            print(kernel)
        kernel = kernel / (16 ** (size // 2))

        return convolve(image, kernel)

    def laplace(self, _type, normalize=False):
        if _type==4:
            kernel =  np.array([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]])
        if _type==8:
            kernel =  np.array([[-1.,-1.,-1.],[-1.,8.,-1.],[-1.,-1.,-1.]])
        if normalize:
            kernel /= np.sum(np.abs(kernel))
        return kernel
    
    # def dog(self, size,fs=1,cs=2):
    #     return self.gaussiano(size,fs)-self.gaussiano(size,cs)

    def high_pass(low_pass):
        def identity_kernel(s):
            kernel = np.zeros(s)
            kernel[s[0]//2,s[1]//2] = 1.
            return kernel
        return identity_kernel(low_pass.shape) - low_pass
    
    def sobel(self, orientacion):
        match orientacion:
            case "Pasabajos llano 3x3":
                print(f"Kernel", kernel)
                
            case "Pasabajos llano 5x5":
                print(f"Kernel", kernel)

    
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
        plt.imshow(conv, cmap='gray')
        plt.show()
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
            #im = np.clip(im/255,0.,1.)
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
                case "Gaussiano 5x5":
                    kernel = self.gaussiano(5)
                    self.convolucion(im, kernel)
                case "Gaussiano 7x7":
                    kernel = self.gaussiano(7)
                    self.convolucion(im, kernel)
                case "Pasaaltos Laplaciano v4":
                    kernel = self.laplace(4)
                    self.convolucion(im, kernel)
                case "Pasaaltos Laplaciano v8":
                    kernel = self.laplace(8)
                    self.convolucion(im, kernel)
                case "Pasabanda Dog 5x5":
                    kernel = self.dog(5)
                    self.convolucion(im, kernel)
                # Sobel Oeste
                case "Sobel O":
                    kernel = np.array([[ 1,  0, -1],
                                    [ 2,  0, -2],
                                    [ 1,  0, -1]])
                    self.convolucion(im, kernel)
                # Sobel Norte
                case "Sobel N":
                    kernel = np.array([[ 1,  2,  1],
                                    [ 0,  0,  0],
                                    [-1, -2, -1]])
                    self.convolucion(im, kernel)
                # Sobel Este
                case "Sobel E":
                    kernel = np.array([[-1,  0,  1],
                                    [-2,  0,  2],
                                    [-1,  0,  1]])
                    self.convolucion(im, kernel)

                # Sobel Sur
                case "Sobel S":
                    kernel = np.array([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]])
                    self.convolucion(im, kernel)

                # Sobel Noroeste (NO)
                case "Sobel NO":
                    kernel = np.array([[ 2,  1,  0],
                                    [ 1,  0, -1],
                                    [ 0, -1, -2]])
                    self.convolucion(im, kernel)

                # Sobel Noreste (NE)
                case "Sobel NE":
                    kernel = np.array([[ 0,  1,  2],
                                    [-1,  0,  1],
                                    [-2, -1,  0]])
                    self.convolucion(im, kernel)

                # Sobel Suroeste (SO)
                case "Sobel SO":
                    kernel = np.array([[ 0, -1, -2],
                                    [ 1,  0, -1],
                                    [ 2,  1,  0]])
                    self.convolucion(im, kernel)

                # Sobel Sureste (SE)
                case "Sobel SE":
                    kernel = np.array([[-2, -1,  0],
                                    [-1,  0,  1],
                                    [ 0,  1,  2]])
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
