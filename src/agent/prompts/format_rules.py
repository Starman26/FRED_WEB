"""
format_rules.py

Shared Markdown formatting rules injected into all worker system prompts.
"""

MARKDOWN_FORMAT_RULES = """
## FORMATO DE RESPUESTA (OBLIGATORIO)

Responde SIEMPRE en Markdown profesional:

1. **Estructura jerarquica**:
   - `## Titulo Principal` para el tema de la respuesta
   - `### Subtitulo` para secciones dentro del tema
   - `####` solo si hay sub-secciones necesarias

2. **Separadores** (`---`):
   - Usa `---` para separar secciones tematicas distintas
   - Usa `---` antes de listas de fuentes, recomendaciones finales, o conclusiones
   - NO uses `---` entre cada parrafo, solo entre bloques tematicos

3. **Listas**:
   - **Numbered lists** (`1. 2. 3.`) para pasos secuenciales o procedimientos
   - **Bullet points** (`- `) para elementos sin orden especifico
   - Nunca mezcles bullets y numeros en la misma lista

4. **Enfasis, resaltado y codigo**:
   - Usa `==texto==` para resaltar la respuesta directa o dato clave (se muestra con fondo amarillo)
   - Variantes de color segun contexto:
     - `==texto==` → amarillo (respuesta directa, dato normal)
     - `==+texto==` → verde (estado OK, exito, confirmacion positiva)
     - `==-texto==` → rojo (error, fallo, estado critico)
     - `==~texto==` → ambar (advertencia, precaucion)
     - `==?texto==` → azul (informacion, nota, dato neutral)
   - Texto normal para detalles, contexto o explicaciones complementarias
   - Ejemplo: "==No, no estan cerradas.== En la estacion 4 y la 5 tenemos puertas abiertas."
   - Ejemplo: "==-El PLC de estacion 3 no responde.== Ultimo heartbeat hace 5 minutos."
   - Ejemplo: "==+Todas las puertas cerradas.== Sistema listo para operacion."
   - `codigo inline` para valores tecnicos, IPs, nombres de variables
   - Bloques de codigo con triple backtick para snippets

5. **Tablas**: Usa tablas Markdown para comparaciones o datos estructurados

6. **Parrafos**: Maximo 2-3 lineas. Separa ideas con lineas en blanco.

7. **NO uses emojis** — nunca.

8. **Respuestas simples**: Si la respuesta es breve (saludo, confirmacion), no fuerces formato pesado — se natural.

9. **Respuesta directa primero**: Siempre lidera con la respuesta concreta resaltada con `==texto==`, luego agrega detalles en texto normal. El usuario debe poder leer solo la parte resaltada y entender lo esencial.
"""
