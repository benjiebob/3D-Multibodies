from argparse import Namespace

SPIN_TRAINOPTIONS = {}
SPIN_TRAINOPTIONS['img_res'] = 224
SPIN_TRAINOPTIONS['rot_factor'] = 30
SPIN_TRAINOPTIONS['noise_factor'] = 0.4
SPIN_TRAINOPTIONS['scale_factor'] = 0.25
SPIN_TRAINOPTIONS = Namespace(**SPIN_TRAINOPTIONS)