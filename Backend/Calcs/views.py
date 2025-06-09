from urllib.parse import unquote
from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from mpmath import zeta, jtheta, pi, exp
from scipy.fft import fft, ifft # Complex function examples
from scipy.special import expi
import re
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from sympy import (
    Integer, im, sympify, lambdify, symbols, I, sin, cos, log, exp, integrate, symbols, solve, singularities, S,
    gamma, lowergamma, uppergamma, polygamma, loggamma, digamma,
    trigamma, multigamma, dirichlet_eta, zeta, lerchphi, polylog, summation, symbols, Sum, oo, mobius, li, pi
)
import sympy
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from django.http import JsonResponse
from rest_framework.decorators import api_view
from Utils.openAIService import infoFunctionGPT;

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
    input_function = unquote(input_function)
    try:
        x, y= symbols('x y', real=True)  # Fuerza a que x e y sean reales
        k = symbols('k', integer=True)
        z = x + I * y

        local_dict = {'pi':pi, '·': '*','I': I,'𝑦': y,'𝑧':z, '𝑥': x,'x': x, 'k': k,'y': y, 'z': z, 'e': np.e,'𝑒': np.e,'exp': exp, 'sIn': sin, 'cos': cos, 'Integral': integrate, 'lI': li,'mobIus': mobius, 'Sum': Sum, 'oo':oo, 'Integer': Integer}
        input_function = input_function.replace("i", "I")  # Corrige la notación imaginaria
        input_function = input_function.replace("𝜋", "pi")  # Corrige la notación imaginaria
        input_function = input_function.replace("·", "*")  # Corrige la notación imaginaria
        input_function = input_function.replace("^", "**")  # Corrige la notación imaginaria
        # input_function = input_function.replace("%F0%9D%9C%8B", "𝜋")  # Corrige la notación imaginaria
        # Parseamos la expresión evitando conversiones incorrectas
        # import pdb; pdb.set_trace()
        expr = parse_expr(input_function, local_dict=local_dict, evaluate=False)
        print(expr)
        # Extraer la parte real e imaginaria
        real_part, imag_part = expr.as_real_imag()

        print(f"Parte real: {real_part}, Parte imaginaria: {imag_part}")

        # Convertir a funciones evaluables
        f_real = lambdify((x, y), real_part, 'numpy')
        f_imag = lambdify((x, y), imag_part, 'numpy')


    except Exception as e:
        return JsonResponse({"error": f"Invalid mathematical expression: {str(e)}"}, status=400)

    # Crear dominio
    domain_x = np.linspace(-2, 2, 100)
    domain_y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(domain_x, domain_y)

    try:
        Z_real = f_real(X, Y)
        Z_imag = f_imag(X, Y)
        
        # if np.allclose(Z_imag, 0, atol=1e-10):
        #     return JsonResponse({
        #         "x": domain_x.tolist(),
        #         "y": domain_y.tolist(),
        #         "magnitude": Z_real.tolist(),  # Se devuelve solo la parte real
        #         "type": "real_only"
        #     })
        # else:
        Z = Z_real + 1j * Z_imag
        magnitude = np.abs(Z)
        phase = np.angle(Z)

        return JsonResponse({
            "x": domain_x.tolist(),
            "y": domain_y.tolist(),
            "magnitude": magnitude.tolist(),
            "phase": phase.tolist(),
            "type": "complex"
        })

    except Exception as e:
        return JsonResponse({"error": f"Error evaluating function: {str(e)}"}, status=400)

@api_view(['GET'])
def analyzeFunction(request):
    expr_str = request.GET.get('expression')
    if not expr_str:
        return Response({'error': 'No expression provided'}, status=400)

    z = symbols('z')
    try:
        expr = sympify(expr_str)
        zeros = solve(expr, z)
        poles = list(singularities(expr, z))
        info = {
            'expression': expr_str,
            'zeros': [str(zero) for zero in zeros],
            'poles': [str(pole) for pole in poles],
            'notes': ''
        }

        if S(0) in zeros:
            info['notes'] += "La función se anula en z=0. "

        if poles:
            if(len(poles)) > 1:
                info['notes'] += f"Tiene {len(poles)} singularidades (polos). "
            else:
                info['notes'] += f"Tiene {len(poles)} singularidad (polos). "

        return Response(info)

    except Exception as e:
        return Response({'error': f'No se pudo analizar la función: {str(e)}'}, status=400)
# Create de views for calculate de complex function
@api_view(['GET'])
def calculateFunction(request):
    print(request)
    input_function = request.GET.get('function', '1x + 1iy')


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
    x = np.linspace(0, 1, 100)  # Parte real
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

@api_view(['GET'])
def analizarFuncionGPT(request):
    expresion = request.GET.get('expression')
    curso = request.GET.get("curso")
    if not expresion:
        return Response({'error': 'No se ha proporcionado ninguna expresión'}, status=400)
    
    try:
        info_generada = infoFunctionGPT(expresion, curso)
        return Response({'expresion': expresion, 'analisis_gpt': info_generada})
    except Exception as e:
        return Response({'error': str(e)}, status=500)