import numpy as np #libreria para manejar arreglos de forma mas eficiente

class SimpleLinearRegression: #clase para el modelo de regresion lineal simple 
    def __init__(self, x, y): #constructor de la clase, este se ejecuta cuando creamos un objeto de la clase, este recibe (x e y)
       
        self.x = np.array(x)    #uso np para convertirlos en arreglos de numpy(np), y los guarda en el objeto.
        self.y = np.array(y)
        self.beta_0 = 0  #estas son las dos variables donde guardamos los 2 parametros que calcularemos
        self.beta_1 = 0
        self._calculate_parameters()

    def _calculate_parameters(self):            #aqui estan los pasos para calcular la regresion 

        n = len(self.x)   #cantidad de datos 
        sum_x = np.sum(self.x)  #suma de los valores de x
        sum_y = np.sum(self.y)  #suma de los valores de y
        sum_x_squared = np.sum(self.x ** 2)  #suma de los cuadrados de cada valor de x
        sum_xy = np.sum(self.x * self.y) #suma del producto de x e y para cada par

        # Cálculo de beta_1 y beta_0
        self.beta_1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)  #pendiente 
        self.beta_0 = (sum_y - self.beta_1 * sum_x) / n                                #interseccion 

    def predict(self, x_value):

        return self.beta_0 + self.beta_1 * x_value  #prediccion del valor de y para un valor dado en x

    def get_regression_equation(self): #muestra la ecuacion de la regresion completa 
    
        return f"y = {self.beta_0:.2f} + {self.beta_1:.2f} * x"   #devuelve un string con la ecuacion de la regresion 

# Dataset basado en el caso Benetton
# x = [23, 26, 30, 34, 43, 48, 52, 57, 58]
# y = [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]
x = [1,2,3,4,5,6,7,8,9,]  # Valores de Advertising (en millones de euros)
y = [2,4,6,8,10,12,14,16,18]  # Valores de Sales (en millones de euros)

# Creación del modelo y cálculo de parámetros
model = SimpleLinearRegression(x, y)

# Imprime la ecuación de regresión
print("Ecuación de regresión:", model.get_regression_equation())

# Solicita un valor al usuario para hacer una predicción
x_value = float(input("Ingresa un valor de Advertising para predecir las Sales: "))
print("Predicción de Sales:", model.predict(x_value))
