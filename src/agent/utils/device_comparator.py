"""
device_comparator.py - Compare actual device state vs expected state.
"""
import math
from typing import Dict, Any


def compare_device(device_type: str, actual: dict, expected: dict, tolerance: dict = None) -> dict:
    """Dispatch to the correct comparator by device_type."""
    tolerance = tolerance or {}
    comparators = {
        "xarm": compare_xarm,
        "abb": compare_abb,
        "plc": compare_plc,
    }
    comparator = comparators.get(device_type.lower())
    if not comparator:
        return {
            "passed": False,
            "score": 0.0,
            "errors": [f"Unknown device type: {device_type}"],
            "details": {},
        }
    return comparator(actual, expected, tolerance)


def compare_xarm(actual: dict, expected: dict, tolerance: dict) -> dict:
    """Compare xArm state: position, joints, gripper."""
    errors = []
    details = {}
    scores = []

    pos_tol = tolerance.get("position_mm", 5.0)
    joint_tol = tolerance.get("joint_deg", 2.0)

    if actual.get("error_code", 0) != 0:
        errors.append(f"Robot in error state: code {actual['error_code']}")
        return {"passed": False, "score": 0.0, "errors": errors, "details": details}

    actual_pos = actual.get("position", {})
    expected_pos = expected.get("target_position", {})
    if actual_pos and expected_pos:
        distance = _euclidean_distance(actual_pos, expected_pos)
        details["position_error_mm"] = round(distance, 2)

        if distance <= pos_tol:
            scores.append(1.0 - (distance / pos_tol) * 0.5)
        else:
            errors.append(f"Position off by {distance:.1f}mm (tolerance: {pos_tol}mm)")
            scores.append(max(0.0, 1.0 - distance / (pos_tol * 3)))

    joint_errors = _compare_joints(actual.get("joints", []), expected.get("target_joints", []), joint_tol)
    if joint_errors is not None:
        details["joints_ok"] = len(joint_errors) == 0
        details["joint_errors"] = joint_errors
        if joint_errors:
            errors.extend(joint_errors)
            scores.append(0.5)
        else:
            scores.append(1.0)

    if "gripper_expected" in expected:
        gripper_ok = actual.get("gripper") == expected["gripper_expected"]
        details["gripper_ok"] = gripper_ok
        if not gripper_ok:
            errors.append(f"Gripper: expected {'closed' if expected['gripper_expected'] else 'open'}")
            scores.append(0.0)
        else:
            scores.append(1.0)

    overall_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "passed": len(errors) == 0,
        "score": round(overall_score, 3),
        "errors": errors,
        "details": details,
    }


def compare_abb(actual: dict, expected: dict, tolerance: dict) -> dict:
    """Compare ABB state: position, orientation, joints, I/O."""
    errors = []
    details = {}
    scores = []

    pos_tol = tolerance.get("position_mm", 5.0)
    orient_tol = tolerance.get("orientation_deg", 2.0)
    joint_tol = tolerance.get("joint_deg", 2.0)

    actual_pos = actual.get("position", {})
    expected_pos = expected.get("target_position", {})
    if actual_pos and expected_pos:
        distance = _euclidean_distance(actual_pos, expected_pos)
        details["position_error_mm"] = round(distance, 2)
        if distance > pos_tol:
            errors.append(f"Position off by {distance:.1f}mm")
            scores.append(max(0.0, 1.0 - distance / (pos_tol * 3)))
        else:
            scores.append(1.0 - (distance / pos_tol) * 0.5)

    actual_orient = actual.get("orientation", {})
    expected_orient = expected.get("target_orientation", {})
    if actual_orient and expected_orient:
        dot = sum(actual_orient.get(k, 0) * expected_orient.get(k, 0) for k in ["q1", "q2", "q3", "q4"])
        dot = min(1.0, max(-1.0, abs(dot)))
        angle_deg = math.degrees(2 * math.acos(dot))
        details["orientation_error_deg"] = round(angle_deg, 2)
        if angle_deg > orient_tol:
            errors.append(f"Orientation off by {angle_deg:.1f} deg")
            scores.append(0.5)
        else:
            scores.append(1.0)

    joint_errors = _compare_joints(actual.get("joints", []), expected.get("target_joints", []), joint_tol)
    if joint_errors is not None:
        details["joints_ok"] = len(joint_errors) == 0
        if joint_errors:
            errors.extend(joint_errors)
            scores.append(0.5)
        else:
            scores.append(1.0)

    io_errors = _compare_io(actual.get("io", {}), expected.get("expected_io", {}))
    if io_errors is not None:
        details["io_ok"] = len(io_errors) == 0
        if io_errors:
            errors.extend(io_errors)
            scores.append(0.0)
        else:
            scores.append(1.0)

    overall_score = sum(scores) / len(scores) if scores else 0.0
    return {"passed": len(errors) == 0, "score": round(overall_score, 3), "errors": errors, "details": details}


def compare_plc(actual: dict, expected: dict, tolerance: dict) -> dict:
    """Compare PLC state: registers, digital I/O, analog I/O, bits."""
    errors = []
    details = {}
    scores = []

    analog_tol = tolerance.get("analog_percent", 5.0)

    dio_errors = _compare_io(actual.get("digital_io", {}), expected.get("expected_digital_io", {}), "Digital I/O")
    if dio_errors is not None:
        details["digital_io_ok"] = len(dio_errors) == 0
        if dio_errors:
            errors.extend(dio_errors)
            scores.append(0.0)
        else:
            scores.append(1.0)

    reg_errors = _compare_io(actual.get("registers", {}), expected.get("expected_registers", {}), "Register")
    if reg_errors is not None:
        details["registers_ok"] = len(reg_errors) == 0
        if reg_errors:
            errors.extend(reg_errors)
            scores.append(0.0)
        else:
            scores.append(1.0)

    expected_aio = expected.get("expected_analog_io", {})
    actual_aio = actual.get("analog_io", {})
    if expected_aio:
        aio_errors = []
        for key, exp_val in expected_aio.items():
            act_val = actual_aio.get(key)
            if act_val is None:
                aio_errors.append(f"Analog I/O {key}: not present")
                continue
            if exp_val != 0:
                pct_error = abs(act_val - exp_val) / abs(exp_val) * 100
            else:
                pct_error = abs(act_val) * 100
            if pct_error > analog_tol:
                aio_errors.append(f"Analog {key}: expected {exp_val}, got {act_val} ({pct_error:.1f}% off)")
        details["analog_io_ok"] = len(aio_errors) == 0
        if aio_errors:
            errors.extend(aio_errors)
            scores.append(0.5)
        else:
            scores.append(1.0)

    bit_errors = _compare_io(actual.get("bits", {}), expected.get("expected_bits", {}), "Bit")
    if bit_errors is not None:
        details["bits_ok"] = len(bit_errors) == 0
        if bit_errors:
            errors.extend(bit_errors)
            scores.append(0.0)
        else:
            scores.append(1.0)

    overall_score = sum(scores) / len(scores) if scores else 0.0
    return {"passed": len(errors) == 0, "score": round(overall_score, 3), "errors": errors, "details": details}


def _euclidean_distance(a: dict, b: dict) -> float:
    """3D Euclidean distance between two position dicts with x, y, z keys."""
    dx = a.get("x", 0) - b.get("x", 0)
    dy = a.get("y", 0) - b.get("y", 0)
    dz = a.get("z", 0) - b.get("z", 0)
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def _compare_joints(actual_joints: list, expected_joints: list, tol_deg: float) -> list | None:
    """Compare joint arrays. Returns list of error strings, or None if no joints to compare."""
    if not actual_joints or not expected_joints:
        return None
    joint_errors = []
    for i, (aj, ej) in enumerate(zip(actual_joints, expected_joints)):
        diff = abs(aj - ej)
        if diff > tol_deg:
            joint_errors.append(f"Joint {i + 1}: off by {diff:.1f} deg")
    return joint_errors


def _compare_io(actual_io: dict, expected_io: dict, label: str = "I/O") -> list | None:
    """Compare I/O or register dicts. Returns list of error strings, or None if nothing to compare."""
    if not expected_io:
        return None
    io_errors = []
    for key, exp_val in expected_io.items():
        act_val = actual_io.get(key)
        if act_val != exp_val:
            io_errors.append(f"{label} {key}: expected {exp_val}, got {act_val}")
    return io_errors
