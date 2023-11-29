from collections import UserDict
from typing import Optional


class IdHolder(UserDict[str, int]):
    def __getitem__(self, person_name: str) -> int:
        """Returns id associated to `person_name`.
        Creates a new unique id in case association has not been made."""
        if person_name not in self.data:
            i = len(self.data)
            assert i not in self.data.values()
            self.data[person_name] = i
        return self.data[person_name]

    def get_reverse(
        self, key: int, default: Optional[str] = None
    ) -> Optional[str]:  # sourcery skip: use-next
        for k, v in self.data.items():
            if v == key:
                return k
        return default
