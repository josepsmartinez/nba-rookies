from os import PathLike
from pathlib import Path


class PeopleDatabase:
    root_dir: Path
    people: dict[str, list[Path]]

    def __init__(self, root_dir: str | PathLike[str]) -> None:
        self.root_dir = Path(root_dir)

        self.people = {}
        for person_name_path in self.root_dir.iterdir():
            person_name = person_name_path.stem
            self.people[person_name] = list(person_name_path.iterdir())

    def __repr__(self) -> str:
        return '\n'.join(
            f'{person_name} has {len(self.people[person_name])} pictures'
            for person_name in self.people
        )


if __name__ == '__main__':
    print('\n'.join((
        f"Base:\n{PeopleDatabase(Path('.') / 'data' / 'people' / 'base')}",
        f"Query:\n{PeopleDatabase(Path('.') / 'data' / 'people' / 'query')}"
    )))
