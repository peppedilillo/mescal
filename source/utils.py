from source.paths import VERPATH


def get_version():
    with open(VERPATH, "r", encoding="utf-16") as f:
        version = f.readline()
    return version
