"""
symptom_map.py
Maps predicted classes to common symptoms.
"""

SYMPTOM_MAP = {

    "Aphids": [
        "small insects on cotton leaves",
        "sticky honeydew on leaves",
        "leaf curling and yellowing",
        "clusters of aphids on leaf underside"
    ],

    "Army worm": [
        "holes in cotton leaves",
        "ragged leaf edges",
        "caterpillars feeding on leaves"
    ],

    "Bacterial Blight": [
        "angular leaf spots",
        "water soaked lesions",
        "yellow halo around spots"
    ],

    "Powdery Mildew": [
        "white powder on leaves",
        "leaf curling",
        "fungal growth on leaf surface"
    ],

    "Target spot": [
        "brown circular lesions",
        "target shaped leaf spots",
        "leaf blight patches"
    ],

    "Healthy": [
        "green healthy leaves",
        "no disease symptoms"
    ]
}