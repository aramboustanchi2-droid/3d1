"""
Professional Acoustic Analysis System

This module is designed for the analysis and detection of acoustic elements in
architectural drawings. It can:
- Detect acoustic spaces (e.g., conference halls, studios, classrooms).
- Analyze sound insulation and absorption materials.
- Calculate reverberation time (RT60).
- Check against acoustic standards.
- Analyze sound pressure levels and ambient noise.

Author: CAD 3D Converter Team
Date: 2025-11-18
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import ezdxf
from ezdxf.document import Drawing
from ezdxf.layouts import Modelspace
from ezdxf.entities import LWPolyline, Insert, Circle
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AcousticSpaceType(Enum):
    """Enumeration for types of acoustic spaces."""
    # Halls and Auditoriums
    CONFERENCE_HALL = "conference_hall"
    AUDITORIUM = "auditorium"
    LECTURE_HALL = "lecture_hall"
    CONCERT_HALL = "concert_hall"
    THEATER = "theater"
    CINEMA = "cinema"

    # Studios
    RECORDING_STUDIO = "recording_studio"
    BROADCAST_STUDIO = "broadcast_studio"
    MUSIC_STUDIO = "music_studio"
    CONTROL_ROOM = "control_room"
    VOCAL_BOOTH = "vocal_booth"

    # Educational
    CLASSROOM = "classroom"
    LANGUAGE_LAB = "language_lab"
    MUSIC_ROOM = "music_room"
    LIBRARY = "library"

    # Office and Public
    OFFICE = "office"
    MEETING_ROOM = "meeting_room"
    CALL_CENTER = "call_center"
    RESTAURANT = "restaurant"

    # Industrial
    INDUSTRIAL_SPACE = "industrial_space"
    MACHINE_ROOM = "machine_room"

    # Healthcare
    HOSPITAL_ROOM = "hospital_room"
    SURGERY_ROOM = "surgery_room"

    UNKNOWN = "unknown"


class AcousticMaterialType(Enum):
    """Enumeration for types of acoustic materials."""
    # Sound Absorbers
    ABSORBER_FOAM = "absorber_foam"
    ABSORBER_PANEL = "absorber_panel"
    ABSORBER_CEILING = "absorber_ceiling"
    ABSORBER_FABRIC = "absorber_fabric"
    ABSORBER_WOOD = "absorber_wood"

    # Sound Insulators
    INSULATION_WALL = "insulation_wall"
    INSULATION_FLOOR = "insulation_floor"
    INSULATION_CEILING = "insulation_ceiling"
    INSULATION_DOOR = "insulation_door"
    INSULATION_WINDOW = "insulation_window"

    # Sound Diffusers
    DIFFUSER_QRD = "diffuser_qrd"  # Quadratic Residue Diffuser
    DIFFUSER_SKYLINE = "diffuser_skyline"
    DIFFUSER_HEMISPHERE = "diffuser_hemisphere"

    # Bass Traps
    BASS_TRAP_CORNER = "bass_trap_corner"
    BASS_TRAP_PANEL = "bass_trap_panel"


class AcousticStandard(Enum):
    """Enumeration for acoustic standards."""
    ISO_3382 = "ISO 3382"  # Acoustics of rooms measurement standard
    ANSI_S12 = "ANSI S12"  # American standard for noise
    DIN_18041 = "DIN 18041"  # German standard for room acoustics
    WHO_GUIDELINES = "WHO Guidelines"  # World Health Organization guidelines
    BUILDING_CODE = "Building Code"  # National building codes


@dataclass
class AcousticMaterial:
    """
    Stores information about an acoustic material.

    Attributes:
        material_type: The type of the acoustic material.
        location: The (x, y) position of the material.
        dimensions: The (width, height, thickness) of the material.
        absorption_coefficient: Sound absorption coefficient (0-1).
        nrc_rating: Noise Reduction Coefficient.
        stc_rating: Sound Transmission Class.
        thickness_mm: Thickness in millimeters.
        layer: The DXF layer the material was found on.
        coverage_area_m2: The coverage area in square meters.
        properties: A dictionary for additional metadata.
    """
    material_type: AcousticMaterialType
    location: Tuple[float, float]
    dimensions: Tuple[float, float, float]
    absorption_coefficient: float = 0.0
    nrc_rating: float = 0.0
    stc_rating: int = 0
    thickness_mm: float = 0.0
    layer: str = ""
    coverage_area_m2: float = 0.0
    properties: Dict = field(default_factory=dict)


@dataclass
class AcousticSpace:
    """
    Stores information about an acoustic space.

    Attributes:
        space_type: The type of the acoustic space.
        name: The name of the space.
        area_m2: Floor area in square meters.
        volume_m3: Volume of the space in cubic meters.
        height_m: Height of the space in meters.
        boundary: A list of (x, y) tuples defining the space's perimeter.
        rt60_target: Target reverberation time in seconds.
        rt60_actual: Calculated actual reverberation time.
        background_noise_db: Background noise level in dB.
        max_spl_db: Maximum sound pressure level in dB.
        materials: A list of AcousticMaterial objects within the space.
        applicable_standards: A list of relevant acoustic standards.
        acoustic_score: An overall score (0-100) for the space's acoustic quality.
        compliance_status: The compliance status (e.g., 'excellent', 'poor').
        layer: The DXF layer the space was found on.
        properties: A dictionary for additional metadata.
    """
    space_type: AcousticSpaceType
    name: str = ""
    area_m2: float = 0.0
    volume_m3: float = 0.0
    height_m: float = 0.0
    boundary: List[Tuple[float, float]] = field(default_factory=list)

    # Acoustic Parameters
    rt60_target: float = 0.0
    rt60_actual: float = 0.0
    background_noise_db: float = 0.0
    max_spl_db: float = 0.0

    # Installed Materials
    materials: List[AcousticMaterial] = field(default_factory=list)

    # Standards
    applicable_standards: List[AcousticStandard] = field(default_factory=list)

    # Scoring
    acoustic_score: float = 0.0
    compliance_status: str = "unknown"

    layer: str = ""
    properties: Dict = field(default_factory=dict)


@dataclass
class NoiseSource:
    """
    Stores information about a noise source.

    Attributes:
        source_type: The type of the noise source (e.g., 'HVAC').
        location: The (x, y) position of the source.
        sound_power_level_db: The sound power level in dB.
        frequency_range: The frequency range (min_hz, max_hz) of the source.
        operating_hours: The operating hours of the source.
        layer: The DXF layer the source was found on.
    """
    source_type: str
    location: Tuple[float, float]
    sound_power_level_db: float
    frequency_range: Tuple[float, float]
    operating_hours: str = "24/7"
    layer: str = ""


@dataclass
class AcousticAnalysisResult:
    """
    Container for the complete acoustic analysis result.

    Attributes:
        spaces: A list of detected AcousticSpace objects.
        materials: A list of detected AcousticMaterial objects.
        noise_sources: A list of detected NoiseSource objects.
        total_spaces: Total number of detected acoustic spaces.
        total_acoustic_area_m2: Total area of all acoustic spaces.
        total_absorber_area_m2: Total area of all sound-absorbing materials.
        total_insulation_area_m2: Total area of all sound-insulating materials.
        average_acoustic_score: The average acoustic score across all spaces.
        compliant_spaces: Number of spaces meeting compliance criteria.
        non_compliant_spaces: Number of spaces not meeting compliance criteria.
        warnings: A list of warnings generated during analysis.
        recommendations: A list of recommendations for improvement.
    """
    spaces: List[AcousticSpace]
    materials: List[AcousticMaterial]
    noise_sources: List[NoiseSource]

    # Overall Statistics
    total_spaces: int = 0
    total_acoustic_area_m2: float = 0.0
    total_absorber_area_m2: float = 0.0
    total_insulation_area_m2: float = 0.0

    # Acoustic Quality
    average_acoustic_score: float = 0.0
    compliant_spaces: int = 0
    non_compliant_spaces: int = 0

    # Alerts
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class AcousticAnalyzer:
    """
    A professional acoustic analyzer for processing CAD drawings.
    """

    # Reverberation Time (RT60) standards in seconds
    RT60_STANDARDS = {
        AcousticSpaceType.CONFERENCE_HALL: (0.6, 1.0),
        AcousticSpaceType.AUDITORIUM: (0.8, 1.2),
        AcousticSpaceType.LECTURE_HALL: (0.6, 0.9),
        AcousticSpaceType.CONCERT_HALL: (1.5, 2.5),
        AcousticSpaceType.THEATER: (1.0, 1.5),
        AcousticSpaceType.CINEMA: (0.8, 1.2),
        AcousticSpaceType.RECORDING_STUDIO: (0.3, 0.5),
        AcousticSpaceType.BROADCAST_STUDIO: (0.25, 0.4),
        AcousticSpaceType.CONTROL_ROOM: (0.25, 0.35),
        AcousticSpaceType.CLASSROOM: (0.4, 0.7),
        AcousticSpaceType.OFFICE: (0.4, 0.6),
        AcousticSpaceType.MEETING_ROOM: (0.4, 0.6),
        AcousticSpaceType.LIBRARY: (0.5, 0.8),
        AcousticSpaceType.RESTAURANT: (0.6, 1.0),
        AcousticSpaceType.HOSPITAL_ROOM: (0.4, 0.6),
        AcousticSpaceType.SURGERY_ROOM: (0.3, 0.5),
    }

    # Background noise standards in dB(A)
    BACKGROUND_NOISE_STANDARDS = {
        AcousticSpaceType.RECORDING_STUDIO: 20,
        AcousticSpaceType.BROADCAST_STUDIO: 25,
        AcousticSpaceType.CONCERT_HALL: 25,
        AcousticSpaceType.AUDITORIUM: 30,
        AcousticSpaceType.CLASSROOM: 35,
        AcousticSpaceType.OFFICE: 40,
        AcousticSpaceType.MEETING_ROOM: 35,
        AcousticSpaceType.LIBRARY: 30,
        AcousticSpaceType.HOSPITAL_ROOM: 30,
        AcousticSpaceType.SURGERY_ROOM: 25,
    }

    def __init__(self):
        """Initializes the analyzer."""
        self.spaces: List[AcousticSpace] = []
        self.materials: List[AcousticMaterial] = []
        self.noise_sources: List[NoiseSource] = []

    def detect_acoustic_spaces(self, doc: Drawing) -> List[AcousticSpace]:
        """
        Detects acoustic spaces from the drawing.

        Args:
            doc: The ezdxf Drawing object.

        Returns:
            A list of detected AcousticSpace objects.
        """
        spaces = []
        msp = doc.modelspace()

        acoustic_layers = [
            'ACOUSTIC', 'SOUND', 'AUDITORIUM', 'STUDIO', 'HALL',
            'CONFERENCE', 'THEATER', 'CINEMA', 'CLASSROOM', 'LECTURE'
        ]

        for entity in msp:
            try:
                layer_name = entity.dxf.layer.upper()

                if not any(keyword in layer_name for keyword in acoustic_layers):
                    continue

                if isinstance(entity, LWPolyline) and entity.is_closed:
                    points = [(p[0], p[1]) for p in entity.get_points()]
                    if len(points) < 3:
                        continue

                    space_type = self._identify_space_type(layer_name)
                    area = self._calculate_polygon_area(points)

                    # Convert area from mm^2 to m^2 if needed (assuming drawing units are mm)
                    area_m2 = area / 1_000_000.0

                    space = AcousticSpace(
                        space_type=space_type,
                        name=entity.dxf.layer,
                        area_m2=area_m2,
                        boundary=points,
                        layer=entity.dxf.layer
                    )

                    # Estimate volume (assuming a default height of 3 meters)
                    space.height_m = 3.0
                    space.volume_m3 = space.area_m2 * space.height_m

                    space.applicable_standards = [AcousticStandard.ISO_3382, AcousticStandard.BUILDING_CODE]

                    if space_type in self.RT60_STANDARDS:
                        rt60_range = self.RT60_STANDARDS[space_type]
                        space.rt60_target = (rt60_range[0] + rt60_range[1]) / 2

                    if space_type in self.BACKGROUND_NOISE_STANDARDS:
                        space.background_noise_db = self.BACKGROUND_NOISE_STANDARDS[space_type]

                    spaces.append(space)
            except Exception as e:
                logging.warning(f"Could not process entity {entity.dxf.handle} for space detection: {e}")

        self.spaces = spaces
        return spaces

    def detect_acoustic_materials(self, doc: Drawing) -> List[AcousticMaterial]:
        """
        Detects acoustic materials from the drawing.

        Args:
            doc: The ezdxf Drawing object.

        Returns:
            A list of detected AcousticMaterial objects.
        """
        materials = []
        msp = doc.modelspace()

        material_keywords = {
            'ABSORBER': AcousticMaterialType.ABSORBER_PANEL,
            'FOAM': AcousticMaterialType.ABSORBER_FOAM,
            'INSULATION': AcousticMaterialType.INSULATION_WALL,
            'DIFFUSER': AcousticMaterialType.DIFFUSER_QRD,
            'BASS_TRAP': AcousticMaterialType.BASS_TRAP_CORNER,
            'ACOUSTIC_CEILING': AcousticMaterialType.ABSORBER_CEILING,
            'ACOUSTIC_PANEL': AcousticMaterialType.ABSORBER_PANEL,
            'SOUND_INSULATION': AcousticMaterialType.INSULATION_WALL,
        }

        for entity in msp:
            try:
                layer_name = entity.dxf.layer.upper()
                material_type = next((mat_type for keyword, mat_type in material_keywords.items() if keyword in layer_name), None)

                if material_type is None:
                    continue

                location, dimensions = (0, 0), (0, 0, 0)
                if isinstance(entity, Insert):  # Block
                    location = (entity.dxf.insert.x, entity.dxf.insert.y)
                    dimensions = (1000.0, 1000.0, 50.0)  # Default dimensions
                elif isinstance(entity, LWPolyline):
                    points = [(p[0], p[1]) for p in entity.get_points()]
                    if len(points) < 2:
                        continue
                    location = points[0]
                    xs, ys = [p[0] for p in points], [p[1] for p in points]
                    width, height = max(xs) - min(xs), max(ys) - min(ys)
                    dimensions = (width, height, 50.0)
                else:
                    continue

                absorption_coeff = self._get_absorption_coefficient(material_type)
                nrc = self._get_nrc_rating(material_type)
                stc = self._get_stc_rating(material_type)
                coverage_area = (dimensions[0] * dimensions[1]) / 1_000_000.0

                material = AcousticMaterial(
                    material_type=material_type,
                    location=location,
                    dimensions=dimensions,
                    absorption_coefficient=absorption_coeff,
                    nrc_rating=nrc,
                    stc_rating=stc,
                    thickness_mm=dimensions[2],
                    layer=entity.dxf.layer,
                    coverage_area_m2=coverage_area
                )
                materials.append(material)
            except Exception as e:
                logging.warning(f"Could not process entity {entity.dxf.handle} for material detection: {e}")

        self.materials = materials
        return materials

    def detect_noise_sources(self, doc: Drawing) -> List[NoiseSource]:
        """
        Detects noise sources from the drawing.

        Args:
            doc: The ezdxf Drawing object.

        Returns:
            A list of detected NoiseSource objects.
        """
        noise_sources = []
        msp = doc.modelspace()

        noise_keywords = {
            'HVAC': (70, (100, 2000)),
            'MECHANICAL': (75, (50, 5000)),
            'ELEVATOR': (65, (125, 1000)),
            'GENERATOR': (80, (63, 8000)),
            'TRANSFORMER': (70, (100, 1000)),
            'FAN': (75, (200, 2000)),
            'PUMP': (80, (100, 2000)),
        }

        for entity in msp:
            try:
                layer_name = entity.dxf.layer.upper()
                for keyword, (spl, freq_range) in noise_keywords.items():
                    if keyword in layer_name:
                        location = (0, 0)
                        if isinstance(entity, Insert):
                            location = (entity.dxf.insert.x, entity.dxf.insert.y)
                        elif isinstance(entity, Circle):
                            location = (entity.dxf.center.x, entity.dxf.center.y)
                        else:
                            continue

                        source = NoiseSource(
                            source_type=keyword,
                            location=location,
                            sound_power_level_db=spl,
                            frequency_range=freq_range,
                            layer=entity.dxf.layer
                        )
                        noise_sources.append(source)
                        break
            except Exception as e:
                logging.warning(f"Could not process entity {entity.dxf.handle} for noise source detection: {e}")

        self.noise_sources = noise_sources
        return noise_sources

    def calculate_rt60(self, space: AcousticSpace) -> float:
        """
        Calculates Reverberation Time (RT60) using the Sabine formula.
        RT60 = 0.161 * V / A
        V: Volume of the space (m^3)
        A: Total equivalent absorption area (m^2)

        Args:
            space: The AcousticSpace object to analyze.

        Returns:
            The calculated RT60 value in seconds.
        """
        if space.volume_m3 <= 0:
            return 0.0

        total_absorption = sum(m.coverage_area_m2 * m.absorption_coefficient for m in space.materials)

        # If no absorbing materials are found, assume a low default absorption
        if total_absorption == 0:
            total_absorption = space.area_m2 * 0.1  # Assume 10% absorption

        rt60 = (0.161 * space.volume_m3) / total_absorption
        return rt60

    def calculate_acoustic_score(self, space: AcousticSpace) -> float:
        """
        Calculates an acoustic score for a space (0-100).

        Args:
            space: The AcousticSpace object to score.

        Returns:
            A score from 0 to 100.
        """
        score = 100.0

        # Penalty for deviating from RT60 standard
        if space.space_type in self.RT60_STANDARDS:
            rt60_range = self.RT60_STANDARDS[space.space_type]
            rt60_actual = space.rt60_actual
            if rt60_actual < rt60_range[0]:
                score -= 20 * (rt60_range[0] - rt60_actual)
            elif rt60_actual > rt60_range[1]:
                score -= 20 * (rt60_actual - rt60_range[1])

        # Penalty for high background noise
        if space.space_type in self.BACKGROUND_NOISE_STANDARDS:
            max_noise = self.BACKGROUND_NOISE_STANDARDS[space.space_type]
            if space.background_noise_db > max_noise:
                score -= 2 * (space.background_noise_db - max_noise)

        # Penalty for lack of absorbing materials
        if not space.materials:
            score -= 30

        return max(0.0, min(100.0, score))

    def analyze(self, dxf_path: str) -> AcousticAnalysisResult:
        """
        Performs a complete acoustic analysis of a DXF file.

        Args:
            dxf_path: The path to the DXF file.

        Returns:
            An AcousticAnalysisResult object with the full analysis.
        """
        logging.info(f"Starting acoustic analysis for: {dxf_path}")
        try:
            doc = ezdxf.readfile(dxf_path)
        except IOError:
            logging.error(f"Cannot open DXF file: {dxf_path}")
            return AcousticAnalysisResult([], [], [])
        except ezdxf.DXFStructureError:
            logging.error(f"Invalid or corrupt DXF file: {dxf_path}")
            return AcousticAnalysisResult([], [], [])

        spaces = self.detect_acoustic_spaces(doc)
        materials = self.detect_acoustic_materials(doc)
        noise_sources = self.detect_noise_sources(doc)

        for space in spaces:
            space.materials = [m for m in materials if self._point_in_polygon(m.location, space.boundary)]
            space.rt60_actual = self.calculate_rt60(space)
            space.acoustic_score = self.calculate_acoustic_score(space)

            if space.acoustic_score >= 80:
                space.compliance_status = "excellent"
            elif space.acoustic_score >= 60:
                space.compliance_status = "good"
            elif space.acoustic_score >= 40:
                space.compliance_status = "fair"
            else:
                space.compliance_status = "poor"

        total_acoustic_area = sum(s.area_m2 for s in spaces)
        total_absorber_area = sum(m.coverage_area_m2 for m in materials if 'ABSORBER' in m.material_type.name)
        total_insulation_area = sum(m.coverage_area_m2 for m in materials if 'INSULATION' in m.material_type.name)
        avg_score = (sum(s.acoustic_score for s in spaces) / len(spaces)) if spaces else 0.0
        compliant = sum(1 for s in spaces if s.acoustic_score >= 60)
        non_compliant = len(spaces) - compliant

        warnings, recommendations = [], []
        for space in spaces:
            if space.acoustic_score < 60:
                warnings.append(f"Space '{space.name}' needs improvement (Score: {space.acoustic_score:.1f})")
            if space.rt60_actual > 0 and space.space_type in self.RT60_STANDARDS:
                rt60_range = self.RT60_STANDARDS[space.space_type]
                if space.rt60_actual > rt60_range[1]:
                    recommendations.append(
                        f"In '{space.name}', add sound absorbers to reduce RT60 from {space.rt60_actual:.2f}s to < {rt60_range[1]:.2f}s."
                    )
                elif space.rt60_actual < rt60_range[0]:
                    recommendations.append(
                        f"In '{space.name}', reduce sound absorbers to increase RT60 from {space.rt60_actual:.2f}s to > {rt60_range[0]:.2f}s."
                    )

        result = AcousticAnalysisResult(
            spaces=spaces,
            materials=materials,
            noise_sources=noise_sources,
            total_spaces=len(spaces),
            total_acoustic_area_m2=total_acoustic_area,
            total_absorber_area_m2=total_absorber_area,
            total_insulation_area_m2=total_insulation_area,
            average_acoustic_score=avg_score,
            compliant_spaces=compliant,
            non_compliant_spaces=non_compliant,
            warnings=warnings,
            recommendations=recommendations
        )
        logging.info(f"Analysis complete. Found {len(spaces)} spaces.")
        return result

    def export_to_json(self, result: AcousticAnalysisResult, output_path: str):
        """
        Exports the analysis result to a JSON file.

        Args:
            result: The AcousticAnalysisResult to export.
            output_path: The path to the output JSON file.
        """
        data = {
            'summary': {
                'total_spaces': result.total_spaces,
                'total_acoustic_area_m2': result.total_acoustic_area_m2,
                'total_absorber_area_m2': result.total_absorber_area_m2,
                'total_insulation_area_m2': result.total_insulation_area_m2,
                'average_acoustic_score': result.average_acoustic_score,
                'compliant_spaces': result.compliant_spaces,
                'non_compliant_spaces': result.non_compliant_spaces,
            },
            'spaces': [s.__dict__ for s in result.spaces],
            'materials': [m.__dict__ for m in result.materials],
            'noise_sources': [ns.__dict__ for ns in result.noise_sources],
            'warnings': result.warnings,
            'recommendations': result.recommendations,
        }

        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum):
                    return obj.value
                return super().default(obj)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=EnumEncoder, ensure_ascii=False, indent=2)
        logging.info(f"Analysis report exported to {output_path}")

    # --- Helper Methods ---

    def _identify_space_type(self, layer_name: str) -> AcousticSpaceType:
        """Identifies space type from layer name."""
        layer_upper = layer_name.upper()
        type_map = {
            'CONCERT': AcousticSpaceType.CONCERT_HALL, 'MUSIC_HALL': AcousticSpaceType.CONCERT_HALL,
            'AUDITORIUM': AcousticSpaceType.AUDITORIUM, 'AMPHITHEATER': AcousticSpaceType.AUDITORIUM,
            'CONFERENCE': AcousticSpaceType.CONFERENCE_HALL,
            'LECTURE': AcousticSpaceType.LECTURE_HALL,
            'THEATER': AcousticSpaceType.THEATER, 'THEATRE': AcousticSpaceType.THEATER,
            'CINEMA': AcousticSpaceType.CINEMA, 'MOVIE': AcousticSpaceType.CINEMA,
            'RECORDING': AcousticSpaceType.RECORDING_STUDIO, 'STUDIO': AcousticSpaceType.RECORDING_STUDIO,
            'BROADCAST': AcousticSpaceType.BROADCAST_STUDIO,
            'CONTROL': AcousticSpaceType.CONTROL_ROOM,
            'CLASSROOM': AcousticSpaceType.CLASSROOM, 'CLASS': AcousticSpaceType.CLASSROOM,
            'LIBRARY': AcousticSpaceType.LIBRARY,
            'OFFICE': AcousticSpaceType.OFFICE,
            'MEETING': AcousticSpaceType.MEETING_ROOM,
        }
        return next((stype for keyword, stype in type_map.items() if keyword in layer_upper), AcousticSpaceType.UNKNOWN)

    def _calculate_polygon_area(self, points: List[Tuple[float, float]]) -> float:
        """Calculates polygon area using the Shoelace formula."""
        if len(points) < 3:
            return 0.0
        area = 0.5 * abs(sum(p1[0]*p2[1] - p2[0]*p1[1] for p1, p2 in zip(points, points[1:] + [points[0]])))
        return area

    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Checks if a point is inside a polygon using the Ray Casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _get_absorption_coefficient(self, material_type: AcousticMaterialType) -> float:
        """Returns a default absorption coefficient for a material type."""
        coefficients = {
            AcousticMaterialType.ABSORBER_FOAM: 0.85,
            AcousticMaterialType.ABSORBER_PANEL: 0.75,
            AcousticMaterialType.ABSORBER_CEILING: 0.70,
            AcousticMaterialType.ABSORBER_FABRIC: 0.60,
            AcousticMaterialType.ABSORBER_WOOD: 0.40,
            AcousticMaterialType.INSULATION_WALL: 0.50,
            AcousticMaterialType.INSULATION_FLOOR: 0.30,
            AcousticMaterialType.INSULATION_CEILING: 0.50,
            AcousticMaterialType.DIFFUSER_QRD: 0.15,
            AcousticMaterialType.DIFFUSER_SKYLINE: 0.20,
            AcousticMaterialType.BASS_TRAP_CORNER: 0.80,
            AcousticMaterialType.BASS_TRAP_PANEL: 0.75,
        }
        return coefficients.get(material_type, 0.30)

    def _get_nrc_rating(self, material_type: AcousticMaterialType) -> float:
        """Returns a default NRC rating for a material type."""
        nrc_values = {
            AcousticMaterialType.ABSORBER_FOAM: 0.90,
            AcousticMaterialType.ABSORBER_PANEL: 0.80,
            AcousticMaterialType.ABSORBER_CEILING: 0.70,
            AcousticMaterialType.ABSORBER_FABRIC: 0.65,
            AcousticMaterialType.BASS_TRAP_CORNER: 0.85,
        }
        return nrc_values.get(material_type, 0.50)

    def _get_stc_rating(self, material_type: AcousticMaterialType) -> int:
        """Returns a default STC rating for a material type."""
        stc_values = {
            AcousticMaterialType.INSULATION_WALL: 50,
            AcousticMaterialType.INSULATION_FLOOR: 55,
            AcousticMaterialType.INSULATION_CEILING: 50,
            AcousticMaterialType.INSULATION_DOOR: 45,
            AcousticMaterialType.INSULATION_WINDOW: 40,
        }
        return stc_values.get(material_type, 30)


def create_acoustic_analyzer() -> AcousticAnalyzer:
    """Factory function to create an AcousticAnalyzer instance."""
    return AcousticAnalyzer()


if __name__ == "__main__":
    print("🎵 Acoustic Analysis System")
    print("=" * 60)
    print(f"✅ {len(AcousticSpaceType)} space types defined.")
    print(f"✅ {len(AcousticMaterialType)} material types defined.")
    print(f"✅ {len(AcousticStandard)} acoustic standards referenced.")
    print("\n📊 Example RT60 Standards:")
    analyzer = create_acoustic_analyzer()
    for space_type, (min_rt, max_rt) in list(analyzer.RT60_STANDARDS.items())[:5]:
        print(f"   - {space_type.value}: {min_rt}-{max_rt}s")
    print("\n✨ Ready for acoustic analysis!")

    # Example usage:
    # analyzer = create_acoustic_analyzer()
    # try:
    #     analysis_result = analyzer.analyze("path/to/your/drawing.dxf")
    #     analyzer.export_to_json(analysis_result, "acoustic_report.json")
    # except FileNotFoundError:
    #     print("\nError: Example DXF file not found. Please provide a valid path.")
