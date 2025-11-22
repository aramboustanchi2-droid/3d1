"""
University Knowledge Agents - دانشگاه‌های برتر دنیا

ایجنت‌های تخصصی برای اتصال به 10 دانشگاه برتر و یادگیری مداوم
"""

# Top 10 Universities with Free Access to Resources
UNIVERSITIES = {
    "MIT": {
        "name": "Massachusetts Institute of Technology",
        "country": "USA",
        "rank": 1,
        "resources": {
            "opencourseware": {
                "url": "https://ocw.mit.edu",
                "description": "Free lecture notes, exams, and videos from MIT courses",
                "type": "courses",
                "format": ["HTML", "PDF", "Video"]
            },
            "research": {
                "url": "https://dspace.mit.edu",
                "description": "MIT Digital Repository - Research papers and theses",
                "type": "papers",
                "format": ["PDF", "HTML"]
            },
            "ai_lab": {
                "url": "https://www.csail.mit.edu/research",
                "description": "MIT CSAIL - AI and Computer Science research",
                "type": "research",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "AI", "Robotics", "Computer Science", "Architecture",
            "Civil Engineering", "Mechanical Engineering", "Electrical Engineering",
            "Management", "Economics", "Operations Research"
        ]
    },
    
    "Stanford": {
        "name": "Stanford University",
        "country": "USA",
        "rank": 2,
        "resources": {
            "online_courses": {
                "url": "https://online.stanford.edu/free-courses",
                "description": "Free Stanford online courses",
                "type": "courses",
                "format": ["Video", "HTML"]
            },
            "ai_lab": {
                "url": "https://ai.stanford.edu",
                "description": "Stanford AI Lab research and publications",
                "type": "research",
                "format": ["HTML", "PDF"]
            },
            "engineering": {
                "url": "https://engineering.stanford.edu/research",
                "description": "Stanford Engineering research",
                "type": "research",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "AI", "Machine Learning", "Deep Learning", "Computer Vision", "NLP",
            "Software Engineering", "Data Science", "Biomedical Engineering",
            "Management", "Entrepreneurship", "Economics",
            "Sustainability", "Operations Management"
        ]
    },
    
    "Cambridge": {
        "name": "University of Cambridge",
        "country": "UK",
        "rank": 3,
        "resources": {
            "repository": {
                "url": "https://www.repository.cam.ac.uk",
                "description": "Cambridge research repository",
                "type": "papers",
                "format": ["PDF", "HTML"]
            },
            "lectures": {
                "url": "https://www.cam.ac.uk/research",
                "description": "Cambridge research and lectures",
                "type": "courses",
                "format": ["HTML", "PDF"]
            },
            "engineering": {
                "url": "https://www.eng.cam.ac.uk/research",
                "description": "Cambridge Engineering Department",
                "type": "research",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "Engineering", "Mathematics", "Physics", "Computer Science", "Architecture",
            "Aerospace Engineering", "Chemical Engineering", "Biotechnology",
            "Economics", "Quantitative Finance", "Policy Analysis"
        ]
    },
    
    "Oxford": {
        "name": "University of Oxford",
        "country": "UK",
        "rank": 4,
        "resources": {
            "research": {
                "url": "https://ora.ox.ac.uk",
                "description": "Oxford Research Archive",
                "type": "papers",
                "format": ["PDF", "HTML"]
            },
            "podcasts": {
                "url": "https://podcasts.ox.ac.uk",
                "description": "Oxford research podcasts and lectures",
                "type": "courses",
                "format": ["Audio", "Video", "HTML"]
            },
            "materials": {
                "url": "https://www.materials.ox.ac.uk/research",
                "description": "Oxford Materials Department",
                "type": "research",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "Engineering", "Materials Science", "Computer Science", "Mathematics",
            "Energy Systems", "Medicine", "Immunology", "Economics", "Development Studies"
        ]
    },
    
    "Berkeley": {
        "name": "UC Berkeley",
        "country": "USA",
        "rank": 5,
        "resources": {
            "eecs": {
                "url": "https://www2.eecs.berkeley.edu/Research/",
                "description": "Berkeley EECS research",
                "type": "research",
                "format": ["HTML", "PDF"]
            },
            "ai_research": {
                "url": "https://bair.berkeley.edu/blog/",
                "description": "Berkeley AI Research (BAIR) blog",
                "type": "research",
                "format": ["HTML", "PDF"]
            },
            "courses": {
                "url": "https://inst.eecs.berkeley.edu/classes-eecs.html",
                "description": "Berkeley course materials",
                "type": "courses",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "AI", "Machine Learning", "Computer Science", "Engineering", "Architecture",
            "Industrial Engineering", "Electrical Engineering", "Information Systems",
            "Economics", "Public Policy", "Business Analytics"
        ]
    },
    
    "ETH_Zurich": {
        "name": "ETH Zurich",
        "country": "Switzerland",
        "rank": 6,
        "resources": {
            "research": {
                "url": "https://www.research-collection.ethz.ch",
                "description": "ETH Research Collection",
                "type": "papers",
                "format": ["PDF", "HTML"]
            },
            "architecture": {
                "url": "https://arch.ethz.ch/en/research.html",
                "description": "ETH Architecture Department",
                "type": "research",
                "format": ["HTML", "PDF"]
            },
            "civil_engineering": {
                "url": "https://baug.ethz.ch/en/research.html",
                "description": "ETH Civil Engineering research",
                "type": "research",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "Architecture", "Civil Engineering", "Structural Engineering", "Computer Science",
            "Environmental Engineering", "Geospatial Analysis", "Robotics",
            "Energy Engineering", "Economics", "Innovation Management"
        ]
    },
    
    "Caltech": {
        "name": "California Institute of Technology",
        "country": "USA",
        "rank": 7,
        "resources": {
            "authors": {
                "url": "https://authors.library.caltech.edu",
                "description": "Caltech research papers",
                "type": "papers",
                "format": ["PDF", "HTML"]
            },
            "courses": {
                "url": "https://www.caltech.edu/research",
                "description": "Caltech research and courses",
                "type": "courses",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "Physics", "Engineering", "Computer Science", "Mathematics",
            "Planetary Science", "Aerospace", "Quantum Computing", "Bioengineering",
            "Applied Economics"
        ]
    },
    
    "Imperial": {
        "name": "Imperial College London",
        "country": "UK",
        "rank": 8,
        "resources": {
            "spiral": {
                "url": "https://spiral.imperial.ac.uk",
                "description": "Imperial research repository",
                "type": "papers",
                "format": ["PDF", "HTML"]
            },
            "civil_engineering": {
                "url": "https://www.imperial.ac.uk/civil-engineering/research/",
                "description": "Imperial Civil Engineering research",
                "type": "research",
                "format": ["HTML", "PDF"]
            },
            "ai": {
                "url": "https://www.imperial.ac.uk/artificial-intelligence/",
                "description": "Imperial AI research",
                "type": "research",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "Engineering", "AI", "Civil Engineering", "Architecture", "Structural Analysis",
            "Biomedical Engineering", "Materials", "Computational Finance", "Economics",
            "Healthcare Systems", "Risk Management"
        ]
    },
    
    "Carnegie_Mellon": {
        "name": "Carnegie Mellon University",
        "country": "USA",
        "rank": 9,
        "resources": {
            "cs_research": {
                "url": "https://www.cs.cmu.edu/research",
                "description": "CMU Computer Science research",
                "type": "research",
                "format": ["HTML", "PDF"]
            },
            "robotics": {
                "url": "https://www.ri.cmu.edu/publications/",
                "description": "CMU Robotics Institute publications",
                "type": "papers",
                "format": ["PDF", "HTML"]
            },
            "architecture": {
                "url": "https://soa.cmu.edu/research",
                "description": "CMU School of Architecture",
                "type": "research",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "AI", "Robotics", "Computer Science", "Architecture", "Engineering",
            "Human-Computer Interaction", "Cybersecurity", "Data Analytics",
            "Industrial Engineering", "Operations Research", "Economics"
        ]
    },
    
    "TU_Delft": {
        "name": "Delft University of Technology",
        "country": "Netherlands",
        "rank": 10,
        "resources": {
            "repository": {
                "url": "https://repository.tudelft.nl",
                "description": "TU Delft research repository",
                "type": "papers",
                "format": ["PDF", "HTML"]
            },
            "architecture": {
                "url": "https://www.tudelft.nl/en/architecture-and-the-built-environment/research",
                "description": "TU Delft Architecture research",
                "type": "research",
                "format": ["HTML", "PDF"]
            },
            "civil_engineering": {
                "url": "https://www.tudelft.nl/en/ceg/research",
                "description": "TU Delft Civil Engineering research",
                "type": "research",
                "format": ["HTML", "PDF"]
            }
        },
        "focus_areas": [
            "Architecture", "Civil Engineering", "Structural Engineering", "Urban Planning", "AI",
            "Transportation Engineering", "Water Resources", "Sustainable Design",
            "Economics", "Project Management"
        ]
    }
}

# Agent Configuration for each university
AGENT_CONFIG = {
    "scraping": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "timeout": 30,
        "retry_attempts": 3,
        "rate_limit": 2  # seconds between requests
    },
    "learning": {
        "update_frequency": "daily",  # يا "weekly", "monthly"
        "max_documents_per_session": 50,
        "content_types": ["PDF", "HTML", "Video transcripts"],
        "languages": ["en", "fa"]
    },
    "storage": {
        "cache_dir": "university_cache",
        "embeddings_dir": "university_embeddings",
        "max_cache_size_gb": 10
    }
}
