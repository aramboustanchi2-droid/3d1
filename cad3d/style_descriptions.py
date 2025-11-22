"""Architectural style knowledge base for KURDO AI.
Provides concise bilingual descriptions, climate notes, and structural synergy hints.
"""

STYLE_DB = {
    "Modern": {
        "en": "Emphasis on function, clean lines, open plans, large glazing.",
        "fa": "تاکید بر عملکرد، خطوط ساده، پلان‌های باز و سطوح شیشه‌ای بزرگ.",
        "climate": "Needs solar control in hot climates; optimize facade U-values.",
        "structure": "Steel or reinforced concrete frames with curtain walls.",
    },
    "Futuristic": {
        "en": "Advanced materials, parametric surfaces, emphasis on innovation.",
        "fa": "مواد پیشرفته، سطوح پارامتریک، تاکید بر نوآوری و فرم‌های جسورانه.",
        "climate": "Adaptive skins and dynamic shading recommended.",
        "structure": "Hybrid steel-composite cores; possible 3D printed components.",
    },
    "Brutalist": {
        "en": "Raw concrete, monolithic massing, honest tectonics.",
        "fa": "بتن خام، جرم یکپارچه، صداقت سازه‌ای.",
        "climate": "Thermal mass useful; mitigate moisture and surface weathering.",
        "structure": "Reinforced concrete shear walls and waffle slabs.",
    },
    "Organic": {
        "en": "Forms inspired by nature, fluid geometry, biophilic integration.",
        "fa": "فرم‌های الهام‌گرفته از طبیعت، هندسه سیال، ادغام زیست‌دوست.",
        "climate": "Passive ventilation and daylight harvesting integral.",
        "structure": "Glulam timber, shell structures, space frames.",
    },
    "Parametric": {
        "en": "Algorithmically driven geometries, performance-led forms.",
        "fa": "هندسه‌های الگوریتمی، فرم‌های مبتنی بر عملکرد.",
        "climate": "Facade optimization for solar, wind and daylight.",
        "structure": "Complex node-steel space frames or optimized trusses.",
    },
    "Islamic": {
        "en": "Courtyards, iwans, geometric patterns, climatic layering.",
        "fa": "حیاط مرکزی، ایوان، الگوهای هندسی، لایه‌بندی اقلیمی.",
        "climate": "Shaded courts and evaporative cooling enhance comfort.",
        "structure": "Masonry arches, domes, timber roofs; modern RC hybrids.",
    },
    "Neoclassical": {
        "en": "Symmetry, columns, proportion echoing classical orders.",
        "fa": "تقارن، ستون‌ها، تناسبات برگرفته از نظم‌های کلاسیک.",
        "climate": "Deep porticos offer shading; thermal bridging must be managed.",
        "structure": "Steel/RC frames clad with stone or GRC panels.",
    },
    "Traditional/Islamic": {
        "en": "Adapted vernacular elements with passive climate logic.",
        "fa": "عناصر بومی سازگار با منطق اقلیمی و تهویه طبیعی.",
        "climate": "Thick walls and shaded voids stabilize internal temperatures.",
        "structure": "Load-bearing masonry with timber or lightweight vaulted roofs.",
    }
}

def get_style_info(style: str, lang: str = "en"):
    data = STYLE_DB.get(style)
    if not data:
        return None
    return {
        "style": style,
        "description": data.get(lang, data.get("en")),
        "climate": data["climate"],
        "structure_synergy": data["structure"],
    }
