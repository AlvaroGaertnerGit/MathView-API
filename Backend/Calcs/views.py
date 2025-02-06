from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from mpmath import zeta
from scipy.fft import fft, ifft # Complex function examples
import re
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from sympy import (
    im, sympify, lambdify, symbols, I, sin, cos, log, exp, integrate,
    gamma, lowergamma, uppergamma, polygamma, loggamma, digamma,
    trigamma, multigamma, dirichlet_eta, zeta, lerchphi, polylog
)
import sympy

@swagger_auto_schema(
    methods=['get'],
    manual_parameters=[
        openapi.Parameter(
            'function', 
            openapi.IN_QUERY, 
            description="Función en formato 'ax + biy'", 
            type=openapi.TYPE_STRING,
            required=True
        )
    ]
)

# Create de views for calculate de complex function
@api_view(['GET'])
def calculateFunctionParam(request):

    input_function = request.GET.get('function')
    
    pattern = r"^\s*(?:([-+]?\d*\*?x\s*([-+]\s*\d*\*?y)?|[-+]?\d+))?\s*(?:\+\s*I\s*\(\s*([-+]?\d*\*?x\s*([-+]\s*\d*\*?y)?|[-+]?\d+)\s*\))?\s*$"
    # if not re.match(pattern, input_function.replace(" ", "")):
    #     return JsonResponse({"error": "Invalid function format. Expected (ax + by) + I(cx + dy), with possible omissions."}, status=400)
    

    try:
        x, y = symbols('x y')
        z = x + I * y
        
        input_function = input_function.replace("z", "(x + I*y)")
        safe_locals = {  # Funciones seguras
            'I': I, 'x': x, 'y': y, 'z': z,
            'sin': sin, 'cos': cos, 'log': log, 'exp': exp, 'integrate': integrate,
            'gamma': gamma, 'lowergamma': lowergamma, 'uppergamma': uppergamma,
            'polygamma': polygamma, 'loggamma': loggamma, 'digamma': digamma,
            'trigamma': trigamma, 'multigamma': multigamma,
            'dirichlet_eta': dirichlet_eta, 'zeta': zeta, 'lerchphi': lerchphi, 'polylog': polylog
        }
        # Convertir la función en una expresión simbólica
        print(input_function)
        expr = sympify(input_function, locals=safe_locals)
        # Asegurar que siempre tenemos una parte real e imaginaria
        real_part, imag_part = expr.as_real_imag()
        # print(real_part)

        print(f"Parte real: {real_part}, Parte imaginaria: {imag_part}")

        # Convertir a función evaluable
        f_real = lambdify((x, y), real_part, 'numpy')
        f_imag = lambdify((x, y), imag_part, 'numpy')

    except (sympy.SympifyError, ValueError) as e:
        return JsonResponse({"error": f"Invalid mathematical expression: {str(e)}"}, status=400)


        # Crear dominio
    domain_x = np.linspace(-2, 2, 100)
    domain_y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(domain_x, domain_y)

    try:
        Z_real = f_real(X, Y)
        Z_imag = f_imag(X, Y)
        Z = Z_real + 1j * Z_imag  # Ensamblamos la función compleja

        magnitude = np.abs(Z)
        phase = np.angle(Z)

        # Reemplazar valores nulos
        magnitude = np.nan_to_num(magnitude)
        phase = np.nan_to_num(phase)

        return JsonResponse({
            "x": domain_x.tolist(),
            "y": domain_y.tolist(),
            "magnitude": magnitude.tolist(),
            "phase": phase.tolist(),
        })
    except Exception as e:
        return JsonResponse({"error": f"Error evaluating function: {str(e)}"}, status=400)


# Create de views for calculate de complex function
@api_view(['GET'])
def calculateFunction(request):

    print(request)
    input_function = request.GET.get('function', '32x + 89iy')


    pattern = r"(-?\d+)x\s*\+\s*(-?\d+)iy"
    # pattern = r"(-?\d+)y\s*\+\s*(-?\d+)ix"
    match = re.match(pattern, input_function)
    if not match:
        return JsonResponse({"error": "Invalid input format. Use format 'ax + biy'"}, status=400)

    coefficients = [int(match.group(1)), int(match.group(2))]
    print(coefficients)
    
    # Domain
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = coefficients[0] * X + coefficients[1] * 1j *Y # Complex domain

    # Complex function
    try:
        f = Z
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

    # Magnitud and phase
    magnitude = np.abs(f).tolist()
    phase = np.angle(f).tolist()

    magnitude = np.abs(f)
    phase = np.angle(f)

    # Replace null values
    magnitude = np.nan_to_num(magnitude)
    phase = np.nan_to_num(phase)

    # Devuelve los datos en formato JSON
    return JsonResponse({
        "x": x.tolist(),
        "y": y.tolist(),
        "magnitude": magnitude.tolist(),
        "phase": phase.tolist(),
    })
# class CalculateFunction(APIView):

#     def get(self, request):
#         return Response({"message": "Hola desde la API"}, status=status.HTTP_200_OK)
@api_view(['GET'])


def calculateZeta(request):
    # Dominio del plano complejo
    x = np.linspace(-4, 4, 100)  # Parte real
    y = np.linspace(-30, 30, 100)  # Parte imaginaria
    X, Y = np.meshgrid(x, y)  # Crear una cuadrícula
    Z = X + 1j * Y  # Crear valores complejos a partir de x e y

    try:
        f = np.vectorize(lambda z: complex(zeta(z)))(Z)  # Evaluar zeta en cada punto
    except Exception as e:
        print(f"Error al calcular zeta: {e}")
        return JsonResponse({"error": str(e)}, status=400)

    # Calcular magnitud y fase
    magnitude = np.abs(f)
    phase = np.angle(f)

    # Reemplazar valores NaN o infinitos
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=5.0, neginf=5.0)
    phase = np.nan_to_num(phase)

    # Responder con datos JSON
    return JsonResponse({
        "x": x.tolist(),
        "y": y.tolist(),
        "magnitude": magnitude.tolist(),
        "phase": phase.tolist(),
    })

@api_view(['GET'])
def calculateFourier(response):
    # Domain
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # zeta function
    try:
        f = fft(X)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

    # Magnitud and phase
    magnitude = np.abs(f).tolist()
    phase = np.angle(f).tolist()

    magnitude = np.abs(f)
    phase = np.angle(f)

    # Replace null values
    magnitude = np.nan_to_num(magnitude)
    phase = np.nan_to_num(phase)

    # Devuelve los datos en formato JSON
    return JsonResponse({
        "x": x.tolist(),
        "y": y.tolist(),
        "magnitude": magnitude.tolist(),
        "phase": phase.tolist(),
    })


@swagger_auto_schema(
    methods=['get'],
    manual_parameters=[
        openapi.Parameter(
            'function', 
            openapi.IN_QUERY, 
            description="integer for value d'", 
            type=openapi.TYPE_INTEGER,
            required=True
        )
    ]
)
@api_view(['GET'])
def calculateMandelbrot(request):
    """
    Calcula el conjunto de Mandelbrot con una resolución estándar de 500x500 y devuelve el número de iteraciones por punto.
    """
    d = request.GET.get('d', '2')

    try:
        d = int(d)
    except ValueError:
        return JsonResponse({"error": "El parámetro d debe ser un número entero."}, status=400)

    # Definir la resolución estándar
    resolution = 500
    x_min, x_max, y_min, y_max = -2, 2, -2, 2
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y  # Crear la cuadrícula en el plano complejo
    Z = np.zeros_like(C, dtype=np.complex128)
    iteration = np.zeros(C.shape, dtype=int)

    max_iter = 100  # Número máximo de iteraciones

    # Cálculo del conjunto de Mandelbrot
    for i in range(max_iter):
        mask = np.abs(Z) < 2  # Puntos que aún no han escapado
        Z[mask] = Z[mask]**d + C[mask]
        iteration[mask] += 1

    return JsonResponse({
        "x": x.tolist(),
        "y": y.tolist(),
        "iterations": iteration.tolist()
    })

