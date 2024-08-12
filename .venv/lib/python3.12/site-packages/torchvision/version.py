__version__ = '0.19.0'
git_version = '48b1edffdc6f34b766e2b4bbf23b78bd4df94181'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
