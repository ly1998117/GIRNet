from .GIRNet import GIRNetPseudo, GIRNetFlow, GIRNetSimple


def get_model(name):
    if 'flow' in name:
        return GIRNetFlow
    if 'pseudo' in name:
        return GIRNetPseudo
    if 'simple' in name:
        return GIRNetSimple
