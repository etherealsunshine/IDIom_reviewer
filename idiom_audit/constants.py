AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AMINO_ACIDS)

PROTGPS_COMPARTMENTS = (
    "nuclear_speckle",
    "p-body",
    "pml-bdoy",
    "post_synaptic_density",
    "stress_granule",
    "chromosome",
    "nucleolus",
    "nuclear_pore_complex",
    "cajal_body",
    "rna_granule",
    "cell_junction",
    "transcriptional",
)

TARGET_COMPARTMENTS = ("nucleolus", "chromosome", "p-body", "stress_granule")
DEFAULT_SEED = 33402

SOURCE_BASE_IDP = "base_idp"
SOURCE_RL_PREFIX = "rl_"
SOURCE_SCRAMBLED_PREFIX = "scrambled_"
SOURCE_CHEAP_PREFIX = "cheap_"
