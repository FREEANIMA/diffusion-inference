from .te import TE
from .vae import VAE

def load_external() -> tuple[VAE, TE]:
    vae = VAE()
    te = TE()
    return vae, te