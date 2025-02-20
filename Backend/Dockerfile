# Usa una imagen de Python como base
FROM python:3.10

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de dependencias e instálalas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código del backend al contenedor
COPY . .

# Expone el puerto 8000
EXPOSE 8000

# Comando para ejecutar el backend con Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "Backend.wsgi:application"]
