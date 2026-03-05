"""
formatters.py - Formateo de datos del lab para display al usuario

Compatible con formato RPC (is_active, equipment_count, active_errors)
y formato legacy (plc_status, cobot_status, doors_closed, is_operational).
"""
from typing import Dict


def format_lab_overview_for_display(overview: Dict) -> str:
    """Formatea el resumen del lab para mostrar al usuario"""
    if not overview.get("success"):
        return f"**Error** al obtener estado del laboratorio: {overview.get('error')}"

    lines = []

    lines.append(f"**Estaciones activas:** {overview['stations_online']}/{overview['total_stations']}")

    if overview.get('stations_with_errors', 0) > 0:
        lines.append(f"**Estaciones con problemas:** {overview['stations_with_errors']}")

    if overview.get('active_errors_count', 0) > 0:
        lines.append(f"**Errores activos:** {overview['active_errors_count']}")

    lines.append("\n### Detalle por Estacion\n")
    lines.append("| # | Nombre | Equipos | Errores | Estado |")
    lines.append("|---|--------|---------|---------|--------|")

    for station in overview.get('stations', []):
        has_errors = station.get('active_errors', 0) > 0
        is_active = station.get('is_active', True) if 'is_active' in station else station.get('is_operational', True)
        status = "CON ERRORES" if has_errors else ("OK" if is_active else "INACTIVA")

        name = station.get('name', '')[:25]
        equip_count = station.get('equipment_count', '-')
        error_count = station.get('active_errors', 0)
        error_str = f"**{error_count}**" if error_count > 0 else "0"

        lines.append(
            f"| {station['station_number']} | {name} | "
            f"{equip_count} | {error_str} | {status} |"
        )

    return "\n".join(lines)


def format_station_details_for_display(details: Dict) -> str:
    """Formatea los detalles de una estacion para mostrar al usuario"""
    if not details.get("success"):
        return f"**Error:** {details.get('error')}"

    station = details['station']
    lines = [f"## Estacion {station['number']} — {station['name']}\n"]
    if station.get('location'):
        lines.append(f"**Ubicacion:** {station['location']}")
    if station.get('description'):
        lines.append(f"{station['description']}\n")

    plc = details.get('plc')
    if plc:
        connected = "Si" if plc.get('is_connected') else "No"
        lines.append(f"### PLC: {plc.get('name', 'Unknown')}")
        if plc.get('model'):
            lines.append(f"- **Modelo:** {plc['model']}")
        if plc.get('ip_address'):
            lines.append(f"- **IP:** `{plc['ip_address']}`")
        lines.append(f"- **Conectada:** {connected} | **Modo:** {plc.get('run_mode', '?')} | **Estado:** {plc.get('status', 'unknown')}")
        if plc.get('error_code'):
            lines.append(f"- **Error:** `{plc['error_code']}`")
        lines.append("")

    cobot = details.get('cobot')
    if cobot:
        connected = "Si" if cobot.get('is_connected') else "No"
        lines.append(f"### Cobot: {cobot.get('name', 'Unknown')}")
        if cobot.get('model'):
            lines.append(f"- **Modelo:** {cobot['model']}")
        if cobot.get('ip_address'):
            lines.append(f"- **IP:** `{cobot['ip_address']}`")
        lines.append(f"- **Conectado:** {connected} | **Estado:** {cobot.get('status', 'unknown')}")
        lines.append(f"- **Modo:** {cobot.get('mode', 0)} ({cobot.get('routine', 'idle')})")
        lines.append("")

    sensors = details.get('sensors', [])
    if sensors:
        lines.append("### Sensores\n")
        lines.append("| Sensor | Tipo | Ubicacion | Estado |")
        lines.append("|--------|------|-----------|--------|")
        for sensor in sensors:
            if sensor.get('type') == 'door':
                status = "Cerrada" if sensor.get('triggered') else "**ABIERTA**"
            else:
                status = "Activo" if sensor.get('triggered') else "Inactivo"
            lines.append(f"| {sensor.get('name', '?')} | {sensor.get('type', '?')} | {sensor.get('location', '?')} | {status} |")
        lines.append("")

    errors = details.get('active_errors', [])
    if errors:
        lines.append("### Errores Activos\n")
        for err in errors:
            sev = err.get('severity', '?').upper()
            lines.append(f"- **[{sev}]** `{err.get('error_code', '?')}`: {err.get('error_message', '?')}")
        lines.append("")

    lines.append("---\n")
    ready = details.get('ready_to_operate', False)
    lines.append(f"### Estado Operativo: {'LISTA' if ready else 'NO LISTA'}")
    if not ready:
        rd = details.get('ready_details', {})
        reasons = []
        if not rd.get('doors_closed', True):
            reasons.append("Puertas no cerradas")
        if not rd.get('plc_connected', True):
            reasons.append("PLC no conectada")
        if not rd.get('no_active_errors', True):
            reasons.append("Hay errores activos")
        for r in reasons:
            lines.append(f"- {r}")

    return "\n".join(lines)


def format_errors_for_display(errors_data: Dict) -> str:
    """Formatea los errores para mostrar al usuario"""
    if not errors_data.get("success"):
        return f"**Error:** {errors_data.get('error')}"

    if errors_data.get('total_errors', 0) == 0:
        return "**No hay errores activos** en el laboratorio."

    lines = [f"## Errores Activos ({errors_data['total_errors']})\n"]

    if errors_data.get('critical_count', 0) > 0:
        lines.append(f"**CRITICOS:** {errors_data['critical_count']}\n")

    lines.append("| Estacion | Equipo | Severidad | Codigo | Mensaje | Fecha |")
    lines.append("|----------|--------|-----------|--------|---------|-------|")

    for err in errors_data.get('errors', []):
        sev = err.get('severity', '?').upper()
        station = err.get('station_number', '?')
        equip = err.get('equipment_name', '?')
        code = err.get('error_code', '-')
        msg = err.get('message', err.get('error_message', '?'))
        created = err.get('created_at', '-')
        lines.append(f"| {station} | {equip} | **{sev}** | `{code}` | {msg} | {created} |")

    return "\n".join(lines)
