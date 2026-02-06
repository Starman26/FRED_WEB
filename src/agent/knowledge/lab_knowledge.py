"""
lab_knowledge.py - Conocimiento base del Laboratorio de Manufactura

Este archivo contiene información fundamental sobre el laboratorio que el agente
debe conocer para responder preguntas y contextualizar sus respuestas.

INSTRUCCIONES PARA EDITAR:
- Modifica las constantes según la configuración real de tu laboratorio
- El agente usará esta información en sus respuestas
"""

# ============================================
# INFORMACIÓN GENERAL DEL LABORATORIO
# ============================================

LAB_NAME = "Laboratorio de Manufactura Avanzada"
LAB_DESCRIPTION = """
El Laboratorio de Manufactura Avanzada es una instalación de producción automatizada
que combina robots colaborativos (cobots), controladores lógicos programables (PLCs)
y sistemas de visión artificial para realizar procesos de ensamblaje, inspección
y empaque de productos electrónicos.

El laboratorio está diseñado para demostrar conceptos de Industria 4.0, incluyendo:
- Automatización flexible con cobots
- Control distribuido mediante PLCs Siemens
- Monitoreo en tiempo real
- Integración de sistemas mediante protocolos industriales (OPC-UA, Modbus)
"""

LAB_PURPOSE = """
Propósito principal: Línea de ensamblaje automatizada para productos electrónicos
con capacidad de personalización y producción flexible.
"""

# ============================================
# ROBOTS Y COBOTS
# ============================================

ROBOTS = {
    "ALFREDO": {
        "nombre_completo": "ALFREDO - Autonomous Lightweight Flexible Robot for EDucation and Operations",
        "tipo": "Cobot Industrial",
        "fabricante": "Universal Robots",
        "modelo": "UR5e",
        "ubicacion": "Estación 1 - Ensamblaje Inicial",
        "descripcion": """
ALFREDO es el cobot principal del laboratorio, un brazo robótico colaborativo
de 6 ejes diseñado para trabajar junto a operadores humanos sin necesidad de
jaulas de seguridad. Está equipado con:
- Pinza adaptativa para manipulación de componentes
- Sensor de fuerza/torque para operaciones delicadas
- Cámara de visión integrada para pick & place
        """,
        "capacidades": [
            "Pick and place de componentes",
            "Ensamblaje de precisión",
            "Inspección visual básica",
            "Paletizado",
        ],
        "rutinas_disponibles": {
            1: "Rutina de ensamblaje estándar",
            2: "Rutina de pick & place",
            3: "Rutina de inspección",
            4: "Rutina de paletizado",
        }
    },
    # Agrega más robots aquí según tu laboratorio
    # "ROBOT_2": { ... }
}

# ============================================
# ESTACIONES DE TRABAJO
# ============================================

STATIONS = {
    1: {
        "nombre": "Ensamblaje Inicial",
        "descripcion": "Primera estación de la línea donde se realiza el ensamblaje base del producto",
        "equipos": ["PLC-ST1 (Siemens S7-1200)", "Cobot ALFREDO", "Sensor de puerta"],
        "operaciones": [
            "Recepción de componentes",
            "Ensamblaje de base",
            "Verificación inicial",
        ],
    },
    2: {
        "nombre": "Soldadura",
        "descripcion": "Estación de soldadura automatizada para conexiones electrónicas",
        "equipos": ["PLC-ST2 (Siemens S7-1200)", "Cobot de soldadura", "Sistema de extracción"],
        "operaciones": [
            "Soldadura por puntos",
            "Soldadura por ola (si aplica)",
            "Inspección de juntas",
        ],
    },
    3: {
        "nombre": "Inspección Visual",
        "descripcion": "Control de calidad mediante visión artificial",
        "equipos": ["PLC-ST3 (Siemens S7-1200)", "Sistema de visión", "Cámara industrial"],
        "operaciones": [
            "Inspección de defectos",
            "Verificación de ensamblaje",
            "Lectura de códigos",
        ],
    },
    4: {
        "nombre": "Ensamblaje Final",
        "descripcion": "Completado del producto con componentes finales",
        "equipos": ["PLC-ST4 (Siemens S7-1200)", "Cobot de ensamblaje", "Herramientas automáticas"],
        "operaciones": [
            "Instalación de carcasa",
            "Conexiones finales",
            "Etiquetado",
        ],
    },
    5: {
        "nombre": "Testing",
        "descripcion": "Pruebas funcionales del producto terminado",
        "equipos": ["PLC-ST5 (Siemens S7-1200)", "Banco de pruebas", "Equipo de medición"],
        "operaciones": [
            "Prueba eléctrica",
            "Prueba funcional",
            "Registro de resultados",
        ],
    },
    6: {
        "nombre": "Empaque",
        "descripcion": "Empaquetado y preparación para envío",
        "equipos": ["PLC-ST6 (Siemens S7-1200)", "Cobot de empaque", "Impresora de etiquetas"],
        "operaciones": [
            "Empaque individual",
            "Etiquetado final",
            "Paletizado",
        ],
    },
}

# ============================================
# EQUIPOS Y TECNOLOGÍAS
# ============================================

EQUIPMENT_INFO = {
    "PLCs": {
        "fabricante": "Siemens",
        "modelos": ["S7-1200", "S7-1500"],
        "software": "TIA Portal V17",
        "protocolos": ["Profinet", "OPC-UA", "Modbus TCP"],
        "descripcion": """
Los PLCs (Controladores Lógicos Programables) son el cerebro de cada estación.
Controlan la secuencia de operaciones, manejan señales de sensores y actuadores,
y se comunican con el sistema central de supervisión.
        """,
    },
    "Cobots": {
        "fabricante": "Universal Robots",
        "modelos": ["UR5e", "UR10e"],
        "software": "Polyscope",
        "descripcion": """
Los cobots (robots colaborativos) están diseñados para trabajar de forma segura
junto a operadores humanos. Incluyen sensores de fuerza que detienen el movimiento
si detectan contacto inesperado.
        """,
    },
    "Sensores de Puerta": {
        "tipo": "Interlock de seguridad",
        "funcion": """
Los sensores de puerta son dispositivos de seguridad que detectan si las puertas
de las celdas de trabajo están abiertas o cerradas. Cuando una puerta está abierta,
los cobots no pueden operar por seguridad.
        """,
    },
}

# ============================================
# TERMINOLOGÍA Y FRASES COMUNES
# ============================================

TERMINOLOGY = {
    "ALFREDO": "Robot colaborativo principal del laboratorio (UR5e en Estación 1)",
    "Interlock": "Sistema de seguridad que impide operación cuando hay condiciones inseguras",
    "Rutina": "Programa de movimientos predefinido para el cobot",
    "Modo RUN": "PLC ejecutando programa normalmente",
    "Modo STOP": "PLC detenido, no ejecuta programa",
    "OEE": "Overall Equipment Effectiveness - Eficiencia general del equipo",
    "Celda": "Área de trabajo delimitada de una estación",
    "Pick & Place": "Operación de tomar y colocar componentes",
    "Profinet": "Protocolo de comunicación industrial de Siemens",
    "TIA Portal": "Software de programación para PLCs Siemens",
    "Polyscope": "Software de programación para robots Universal Robots",
    "Teach Pendant": "Control manual para programar movimientos del cobot",
    "TCP": "Tool Center Point - Punto central de la herramienta del robot",
    "Payload": "Carga máxima que puede manipular el robot",
    "Ciclo": "Una ejecución completa de la rutina de producción",
    "Tiempo de ciclo": "Duración de un ciclo completo de producción",
}

# ============================================
# PAPERS Y DOCUMENTACIÓN IMPORTANTE
# ============================================

IMPORTANT_DOCUMENTS = [
    {
        "titulo": "Manual de Operación del Laboratorio",
        "descripcion": "Guía completa de operación y procedimientos estándar",
        "temas": ["Seguridad", "Operación", "Mantenimiento básico"],
    },
    {
        "titulo": "Guía de Programación TIA Portal",
        "descripcion": "Referencia para programación de PLCs Siemens",
        "temas": ["Ladder", "SCL", "Bloques de función"],
    },
    {
        "titulo": "Manual Universal Robots",
        "descripcion": "Documentación técnica de los cobots UR",
        "temas": ["Programación", "Seguridad", "Mantenimiento"],
    },
    # Agrega más documentos relevantes
]

# ============================================
# PROCEDIMIENTOS DE SEGURIDAD
# ============================================

SAFETY_PROCEDURES = """
## Procedimientos de Seguridad del Laboratorio

### Antes de operar:
1. Verificar que todas las puertas de seguridad estén cerradas
2. Confirmar que no hay personal en las celdas de trabajo
3. Verificar estado de los indicadores luminosos (verde = OK)

### Durante la operación:
- No abrir puertas mientras los cobots están en movimiento
- Usar el botón de paro de emergencia ante cualquier situación anormal
- Mantener las manos fuera del área de operación

### En caso de emergencia:
1. Presionar el botón de PARO DE EMERGENCIA más cercano
2. Notificar al supervisor
3. No intentar reiniciar sin autorización
"""

# ============================================
# ERRORES COMUNES Y SOLUCIONES
# ============================================

COMMON_ERRORS = {
    "CONN_TIMEOUT": {
        "descripcion": "Timeout de conexión con el PLC",
        "causas": ["Cable de red desconectado", "PLC apagado", "Problema de red"],
        "solucion": "Verificar conexión física y estado del PLC",
    },
    "DOOR_OPEN": {
        "descripcion": "Puerta de seguridad abierta",
        "causas": ["Puerta físicamente abierta", "Sensor de puerta dañado"],
        "solucion": "Cerrar la puerta o verificar el sensor",
    },
    "COBOT_FAULT": {
        "descripcion": "Falla en el cobot",
        "causas": ["Colisión detectada", "Sobrecarga", "Error de programa"],
        "solucion": "Revisar el teach pendant para ver el error específico",
    },
    "ESTOP_ACTIVE": {
        "descripcion": "Paro de emergencia activado",
        "causas": ["Botón de emergencia presionado"],
        "solucion": "Liberar el botón de emergencia y realizar reset",
    },
}


# ============================================
# FUNCIONES HELPER
# ============================================

def get_lab_knowledge_summary() -> str:
    """Retorna un resumen del conocimiento del laboratorio para incluir en prompts"""
    
    # Construir lista de robots
    robots_list = []
    for name, info in ROBOTS.items():
        robots_list.append(f"- **{name}**: {info['tipo']} - {info['ubicacion']}")
    
    # Construir lista de estaciones
    stations_list = []
    for num, info in STATIONS.items():
        stations_list.append(f"- **Estación {num}** - {info['nombre']}: {info['descripcion'][:80]}...")
    
    return f"""## Conocimiento del Laboratorio

### Descripción General
{LAB_DESCRIPTION.strip()}

### Robots Disponibles
{chr(10).join(robots_list)}

### Estaciones de Trabajo
{chr(10).join(stations_list)}

### Terminología Clave
{chr(10).join([f"- **{k}**: {v}" for k, v in list(TERMINOLOGY.items())[:10]])}
"""


def get_robot_info(robot_name: str) -> str:
    """Obtiene información detallada de un robot específico"""
    robot_name_upper = robot_name.upper()
    
    if robot_name_upper in ROBOTS:
        robot = ROBOTS[robot_name_upper]
        rutinas = "\n".join([f"  - Rutina {k}: {v}" for k, v in robot.get("rutinas_disponibles", {}).items()])
        
        return f"""## {robot_name_upper}
**Nombre completo:** {robot.get('nombre_completo', 'N/A')}
**Tipo:** {robot.get('tipo', 'N/A')}
**Fabricante:** {robot.get('fabricante', 'N/A')}
**Modelo:** {robot.get('modelo', 'N/A')}
**Ubicación:** {robot.get('ubicacion', 'N/A')}

### Descripción
{robot.get('descripcion', 'Sin descripción')}

### Rutinas Disponibles
{rutinas}
"""
    
    return f"No se encontró información sobre el robot '{robot_name}'"


def get_station_info(station_number: int) -> str:
    """Obtiene información detallada de una estación"""
    if station_number in STATIONS:
        station = STATIONS[station_number]
        equipos = "\n".join([f"  - {e}" for e in station.get("equipos", [])])
        operaciones = "\n".join([f"  - {o}" for o in station.get("operaciones", [])])
        
        return f"""## Estación {station_number} - {station.get('nombre', 'N/A')}

### Descripción
{station.get('descripcion', 'Sin descripción')}

### Equipos
{equipos}

### Operaciones
{operaciones}
"""
    
    return f"No se encontró información sobre la estación {station_number}"


def get_terminology_definition(term: str) -> str:
    """Busca la definición de un término"""
    term_upper = term.upper()
    term_lower = term.lower()
    
    # Buscar coincidencia exacta o parcial
    for key, value in TERMINOLOGY.items():
        if key.upper() == term_upper or key.lower() == term_lower:
            return f"**{key}**: {value}"
    
    # Buscar coincidencia parcial
    for key, value in TERMINOLOGY.items():
        if term_lower in key.lower() or term_lower in value.lower():
            return f"**{key}**: {value}"
    
    return f"No se encontró definición para '{term}'"


def get_error_solution(error_code: str) -> str:
    """Obtiene información sobre un error y su solución"""
    error_upper = error_code.upper()
    
    if error_upper in COMMON_ERRORS:
        error = COMMON_ERRORS[error_upper]
        causas = "\n".join([f"  - {c}" for c in error.get("causas", [])])
        
        return f"""## Error: {error_upper}

**Descripción:** {error.get('descripcion', 'N/A')}

**Posibles causas:**
{causas}

**Solución recomendada:** {error.get('solucion', 'Contactar soporte técnico')}
"""
    
    return f"No se encontró información sobre el error '{error_code}'"
