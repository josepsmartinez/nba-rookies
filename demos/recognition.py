from pathlib import Path

from ifrecog import RecognitionEngine, PeopleDatabase

if __name__ == '__main__':
    engine = RecognitionEngine(Path('.') / 'data/people/base')
    query_db = PeopleDatabase(Path('.') / 'data/people/query')

    print(engine.people_db)
    print(query_db)

    for src, result in engine.query_people(query_db).items():
        print(f'{src} => {result}')
    for src, result in engine.query_people(engine.people_db).items():
        print(f'{src} => {result}')
