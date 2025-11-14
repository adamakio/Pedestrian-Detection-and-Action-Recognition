# action_det/heads.py
HEADS = {
    "atomic": [
        "none",
        "standing", "jumping", "squatting", "bending",
        "running", "walking", "laying down", "sitting", "kneeling"
    ],
    "simple-context": [
        "none",
        "motorcycling", "jaywalking", "crossing legally", "entering a building",
        "waiting o cross", "opening", "walking on the side", "cleaning",
        "closing", "exiting a building", "walking on the road", "biking"
    ],
    "complex-context": [
        "none",
        "getting-on 2wv", "getting off 2wv", "getting-out 4wv",
        "getting in 4wv", "loading", "unloading"
    ],
    "communicative": [
        "none",
        "talking on phone", "looking at phone", "talking in group"
    ],
    "transportive": [
        "none",
        "pulling", "carrying", "pushing"
    ],
}

# CSV column -> head key (only these 5)
COL2HEAD = {
    "attributes.Atomic Actions": "atomic",
    "attributes.Simple Context": "simple-context",
    "attributes.Complex Contextual": "complex-context",
    "attributes.Communicative": "communicative",
    "attributes.Transporting": "transportive",
}
