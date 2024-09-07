
from yacs.config import CfgNode as CN

_C = CN()

# Ligand feature extractor
_C.LIGAND = CN()
_C.LIGAND.NODE_IN_FEATS = 75
_C.LIGAND.PADDING = True
_C.LIGAND.HIDDEN_LAYERS = [128, 128, 128]
_C.LIGAND.NODE_IN_EMBEDDING = 128
_C.LIGAND.MAX_NODES = 290

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True
_C.PROTEIN.THERSHOLD = 8
_C.PROTEIN.NUM_CAND = 1
_C.PROTEIN.CHANNEL = 16

_C.PROTEIN.NODE_IN_FEATS = 75
_C.PROTEIN.NODE_IN_EMBEDDING = 128
_C.PROTEIN.HIDDEN_LAYERS = [128, 128, 128]

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2
# Transformer setting
_C.TRANS = CN()
_C.TRANS.IN_DIM = 128
_C.TRANS.HEAD = 2

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 128
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048

# DIR
_C.DIR = CN()
_C.DIR.DATASET = 'dataset path'
_C.DIR.FEATURE = 'pdbfile path'

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "output dir path"
_C.RESULT.SAVE_MODEL = True


# Comet config, ignore it If not installed.
_C.COMET = CN()
# Please change to your own workspace name on comet.
_C.COMET.WORKSPACE = "kelvin"
_C.COMET.PROJECT_NAME = "Trans"
_C.COMET.USE = False
_C.COMET.TAG = None


def get_cfg_defaults():
    return _C.clone()
