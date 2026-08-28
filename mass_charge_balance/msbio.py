"""Back-compat shim: msbio.* now lives in mcb.io. Kept so exploration/validation
scripts that `import msbio` keep working. New code should `from mcb import io`."""
from mcb.io import *  # noqa: F401,F403
from mcb.io import (  # explicit re-export of module-level constants
    BIOCHEM, CPD_PROTON, CPD_WATER, UNKNOWN_SENTINELS,
)
