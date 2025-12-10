import pickle
from pathlib import Path
from typing import Any, Union

def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save an object as a Pickle file. Creates directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load an object from a Pickle file."""
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)



