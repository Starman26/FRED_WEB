"""
lab_profile.py - Contexto base del laboratorio FrED Factory

Este archivo contiene información esencial sobre el laboratorio que SIEMPRE
debe estar disponible para el agente. No requiere búsqueda RAG.

Incluye:
- Identidad del laboratorio
- Descripción del proyecto Al_FrED_0
- Las 6 estaciones y sus funciones
- Equipamiento (cobots, PLCs, sensores)
- Terminología y modismos internos
"""

LAB_PROFILE = """
## 🏭 FrED Factory - Laboratorio de Manufactura Inteligente

### Identidad
La **FrED Factory** es un laboratorio de manufactura inteligente ubicado en el Tecnológico de Monterrey, 
desarrollado en colaboración con el MIT. Es una "learning factory" donde se construye el dispositivo 
**Al_FrED_0** (Fiber Extrusion Device) - un extrusor de filamento educativo.

### El Proyecto Al_FrED_0
**Al_FrED_0** es la versión mexicana del FrED desarrollado originalmente en MIT. Es un dispositivo de 
extrusión de filamento de bajo costo diseñado para educación en ingeniería mecatrónica.

**Áreas de aprendizaje que cubre:**
- Sistemas embebidos (Arduino, Raspberry Pi, ESP32)
- Diseño mecánico y electrónico
- Sistemas de control (PID para temperatura)
- IoT y comunicaciones (WiFi, Bluetooth, Thinger.io)
- Visión por computadora (YOLO para medir diámetro del filamento)
- Manufactura aditiva
- Actuadores (motores stepper, heating cartridge)
- Sensores (termistor, cámara)

El dispositivo ha pasado por **3 generaciones** de evolución y ha impactado a cientos de estudiantes
del Tecnológico de Monterrey a través de clases como Diseño Mecatrónico y Automatización de 
Sistemas de Manufactura.

### Las 6 Estaciones de la FrED Factory

La FrED Factory tiene **6 estaciones colaborativas humano-robot** donde se ensambla el Al_FrED_0.
El proceso es secuencial: cada estación agrega componentes hasta que la estación 6 entrega el 
dispositivo completo y probado.

**Modo de operación colaborativo:**
- 🤖 **Los cobots** levantan, posicionan y sostienen las piezas con precisión
- 👷 **Los operadores humanos** realizan los atornillados y conexiones delicadas
- La **estación 6** deja el Al_FrED_0 completamente ensamblado, probado y listo

| Estación | Nombre | Función | Cobot |
|----------|--------|---------|-------|
| **1** | Ensamblaje Base | Armado de la estructura base y montaje del frame principal de aluminio | UR3e |
| **2** | Sistema de Extrusión | Instalación del extrusor Artillery Sidewinder X1, heating block (20x20x12mm) y nozzle | UR3e |
| **3** | Electrónica Principal | Montaje del Arduino Mega 2560 Rev3 + Ramps 1.4 shield + 2x drivers A4988 | UR5e |
| **4** | Sistema de Control | Instalación de ESP32-D0WD-V3, cableado de potencia y señales, conexión de sensores | UR5e |
| **5** | Visión y Cámara | Montaje de Raspberry Pi 5 + cámara Arducam IMX477 12.3MP + lente microscópico EBTOOLS 8X-100X | UR3e |
| **6** | QA y Finalización | Inspección final, pruebas de funcionamiento, calibración, empaque y liberación | UR10e |

### Equipamiento por Estación

**PLCs (Siemens S7-1200)**
- Cada estación tiene una PLC que coordina el cobot y los sensores
- Comunicación via Profinet
- Red: 192.168.1.10x (donde x = número de estación)
- Programadas en TIA Portal V17

**Cobots (Universal Robots)**
- Estaciones 1, 2, 5: UR3e (carga útil 3kg, alcance 500mm) - para piezas pequeñas
- Estaciones 3, 4: UR5e (carga útil 5kg, alcance 850mm) - para componentes medianos
- Estación 6: UR10e (carga útil 10kg, alcance 1300mm) - para ensamble final y manipulación completa
- Todos operan con Polyscope 5.x

**Sensores de Seguridad**
- Cada estación tiene sensor de puerta (interlock de seguridad)
- Si la puerta está abierta, el cobot NO puede operar
- Sensores conectados a las entradas digitales de la PLC

### Hardware del Al_FrED_0 (lo que se construye)

**Controladores:**
- **Arduino Mega 2560 Rev3** con Ramps 1.4 shield (control principal de actuadores)
- **2x Driver A4988** para motores stepper (microstepping 1/16 = 3200 steps/rev)
- **ESP32-D0WD-V3** (comunicación WiFi/Bluetooth a 16MHz)
- **Raspberry Pi 5** a 1.5GHz (visión por computadora)

**Motores:**
- **2x NEMA 17** (modelo 17HS4401): 12V, 1.7A, step angle 1.8°, 200 steps/rev
- Motor de extrusión conectado a E0 en Ramps 1.4
- Motor de bobinado (spool) conectado al eje X en Ramps 1.4
- Vref recomendado: 0.648V (70% de capacidad)

**Sistema Térmico:**
- **Heating cartridge 64W** (12-24V, 2.67-5.33A)
- **Termistor NTC 3950 100K** conectado a pin A15
- Heating block de aluminio 20x20x12mm
- Control PID con PWM en pin D10
- Setpoint típico: 200°C
- Precisión: error < 1°C
- Tiempo de calentamiento: < 3 minutos

**Ventilación:**
- **Evercool 60mm x 15mm** Dual-Ball PWM Fan (12V, 0.06-0.21A)
- Conectado a pin D9

**Sistema de Visión:**
- Cámara **Arducam IMX477** 12.3MP (Raspberry Pi HQ Camera)
- Lente microscópico **EBTOOLS** CCD Mount 8X-100X
- Detección de diámetro con **YOLO** (antes usaba edge detection)
- Medición en tiempo real del filamento

**Comunicación e IoT:**
- Primera versión: Blynk via ESP32 WiFi
- Versión actual: **Thinger.io** via serial Arduino-Raspberry Pi
- Dashboard remoto con: temperatura real-time, video streaming, logging de diámetros
- El ESP32 fue eliminado en la versión actual para simplificar

### Fórmulas Técnicas Importantes

**Cálculo de temperatura (Steinhart-Hart para termistor NTC 3950):**
```
Rt = R0 × e^(B × (1/T - 1/T0))
T(°C) = 1 / ((1/B) × ln(Rt/R0) + 1/T0) - 273.15

Donde:
- R0 = 100K ohms (resistencia a temperatura de referencia)
- B = 3950 (coeficiente beta)
- T0 = 298.15K (25°C, temperatura de referencia)
```

**Cálculo de Vref para driver A4988:**
```
Vref = Ilimit × 8 × Rcs
Vref = 1.19A × 8 × 0.068Ω ≈ 0.648V

Donde:
- Ilimit = 70% de corriente nominal del motor (1.7A × 0.7 = 1.19A)
- Rcs = 0.068Ω (resistor de sensado del A4988)
```

### Especificaciones Resumidas

| Parámetro | Valor |
|-----------|-------|
| Temperatura máxima | 200°C |
| Precisión temperatura | ±1°C |
| Mejora en tiempo de respuesta | 30% con PID optimizado |
| Tiempo calentamiento | <3 minutos |
| Diámetro filamento objetivo | 1.75mm nominal |
| Voltaje operación | 12-24V DC |
| Corriente máxima sistema | 5A @ 12V |
| Steps por revolución (con microstepping) | 3200 |
"""

TERMINOLOGY = """
## 📚 Terminología y Modismos del Lab

### Nombres de Equipos
- **"El FrED"**, **"Alfred"**, **"El Alfred"**: El dispositivo Al_FrED_0
- **"La fábrica"**, **"el lab"**: La FrED Factory completa
- **"El UR de la 3"**: El cobot UR5e de la estación 3
- **"La Mega"**: El Arduino Mega 2560
- **"La Raspi"**: La Raspberry Pi 5
- **"El ESP"**: El ESP32-D0WD-V3
- **"El Ramps"**: El shield Ramps 1.4
- **"La línea"**: Las 6 estaciones en secuencia
- **"El rack"**: Gabinete con PLCs y switches de red

### Acciones Comunes
- **"Calentar el nozzle"**: Iniciar el control PID de temperatura hasta 200°C
- **"Correr el spool"** / **"Correr bobinado"**: Activar el motor de bobinado
- **"Extruir"**: Activar el motor de extrusión
- **"Checar el diámetro"**: Ver la medición de la cámara con YOLO
- **"Resetear la estación"**: Poner PLC y cobot en estado inicial (home)
- **"Correr rutina"**: Ejecutar programa automático del cobot
- **"Hacer home"**: Llevar el cobot a posición inicial

### Estados del Sistema
- **"En rutina"** / **"En automático"**: Cobot ejecutando programa automático
- **"En espera"** / **"Idle"**: Sistema listo pero sin producir
- **"Parado por interlock"**: Puerta abierta, cobot detenido por seguridad
- **"En modo manual"** / **"En teach"**: Operador controlando el cobot directamente
- **"Calentando"**: Heater activo, esperando llegar a setpoint
- **"En setpoint"**: Temperatura estabilizada en 200°C

### Errores Frecuentes y Códigos
- **CONN_TIMEOUT**: PLC no responde - revisar cable de red o switch
- **INTERLOCK_OPEN**: Puerta de seguridad abierta - cerrar puerta para continuar
- **TEMP_RUNAWAY**: Temperatura fuera de control - apagar heater inmediatamente
- **ESTOP_ACTIVE**: Botón de emergencia presionado - liberar E-STOP y resetear
- **COBOT_FAULT**: Falla en cobot - revisar Polyscope para detalles
- **PLC_DISCONNECTED**: Sin comunicación con PLC - verificar red Profinet

### Clases donde se usa Al_FrED_0
- **Diseño Mecatrónico**: Diseño de componentes, metodologías, normas
- **Automatización de Sistemas de Manufactura**: Implementación, simulación, procesos
- **Internships en campus**: Estudiantes de todos los semestres de mecatrónica
"""

SAFETY_RULES = """
## ⚠️ Reglas de Seguridad

### Reglas Críticas (NUNCA violar)
1. **NUNCA** iniciar un cobot con la puerta de la estación abierta
2. **NUNCA** entrar a la celda mientras el cobot está en movimiento
3. El **botón de emergencia** (E-STOP) rojo detiene TODO inmediatamente - usarlo ante cualquier duda
4. **NUNCA** tocar el nozzle o heating block sin verificar que esté frío (<50°C)
5. **NUNCA** operar el heater sin el ventilador funcionando

### Antes de Operar
- Verificar que NO haya personas en el área de trabajo del cobot
- Confirmar que todas las puertas de seguridad estén cerradas
- Revisar que no haya objetos sueltos en la celda
- Verificar el estado del E-STOP (debe estar liberado)

### Durante la Operación
- Usar lentes de seguridad al trabajar con el extrusor activo
- Mantener distancia del cobot en movimiento
- No distraer al operador que esté controlando el sistema
- Reportar cualquier comportamiento anormal inmediatamente

### Al Terminar
- Esperar a que el nozzle enfríe antes de manipular (<50°C)
- Dejar el cobot en posición home
- Apagar el heater si no se va a usar
- Cerrar las puertas de la estación

### Temperaturas de Precaución
| Componente | Temp. Operación | Peligro |
|------------|-----------------|---------|
| Nozzle | 200°C | Quemadura severa |
| Heating block | 200°C | Quemadura severa |
| Motor NEMA 17 | ~60°C | Quemadura leve |
| Driver A4988 | ~80°C | No tocar |
"""

# Contexto completo para inyectar en prompts
FULL_LAB_CONTEXT = f"""
{LAB_PROFILE}

{TERMINOLOGY}

{SAFETY_RULES}
"""

# Versión corta para no usar muchos tokens en cada llamada
LAB_CONTEXT_SHORT = """
**FrED Factory** - Laboratorio de manufactura del Tecnológico de Monterrey/MIT
donde se construye el **Al_FrED_0** (extrusor de filamento educativo).

**6 Estaciones colaborativas humano-robot:**
| Est. | Función | Cobot |
|------|---------|-------|
| 1 | Ensamblaje Base | UR3e |
| 2 | Sistema de Extrusión (Artillery Sidewinder X1) | UR3e |
| 3 | Electrónica Principal (Arduino Mega + Ramps 1.4) | UR5e |
| 4 | Sistema de Control (ESP32, cableado) | UR5e |
| 5 | Visión y Cámara (Raspberry Pi 5 + Arducam) | UR3e |
| 6 | QA y Finalización | UR10e |

**Modo de trabajo:** Cobots posicionan → Humanos atornillan → Est. 6 entrega listo

**Equipamiento:** PLC Siemens S7-1200 por estación + sensores de puerta (interlock)

**Al_FrED_0 specs:** Control PID temperatura (<1°C error), YOLO para medir diámetro,
Thinger.io para IoT, Arduino Mega + Ramps 1.4, Raspberry Pi 5 para visión.

**Terminología:** "El FrED/Alfred" = dispositivo, "La Mega" = Arduino, 
"La Raspi" = RPi5, "La línea" = las 6 estaciones
"""

def get_lab_context() -> str:
    """Retorna el contexto completo del laboratorio"""
    return FULL_LAB_CONTEXT

def get_lab_context_short() -> str:
    """Retorna versión corta del contexto (menos tokens)"""
    return LAB_CONTEXT_SHORT

def get_lab_profile() -> str:
    """Retorna solo el perfil del laboratorio"""
    return LAB_PROFILE

def get_terminology() -> str:
    """Retorna la terminología"""
    return TERMINOLOGY

def get_safety_rules() -> str:
    """Retorna las reglas de seguridad"""
    return SAFETY_RULES

