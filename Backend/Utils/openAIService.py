from openai import OpenAI
from django.conf import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def infoFunctionGPT(expresion: str, curso: int) -> str:
    if curso == 1:
        cursoStr = "Bacchillerato"
    if curso == 2:
        cursoStr = "ESO"
    if curso == 3:
        cursoStr = "Primaria"
    else :
        cursoStr = "Universitario o estudiante de máster"
    prompt = (
        f"Analiza la siguiente función matemática compleja: {expresion}, tratando z como una variable compleja"
        "Describe su comportamiento, singularidades, dominio, posibles raíces, simetrías, "
        "y cualquier dato relevante que se pueda extraer."
        f"Haz la explicación para un esudiante que está cursando: {cursoStr}"
        "Dame la respuesta en texto plano, si quieres separa en puntos y párrafos, voy a incorporar tal cual el mensaje en el front"
        "No pongas nada más que la información"
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Eres un experto en análisis matemático."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
