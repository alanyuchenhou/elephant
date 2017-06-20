import os

import pandas


def main():
    users = pandas.read_csv(
        os.path.join(os.path.dirname(__file__), '../resources/u.user'),
        sep='|',
        names=['id', 'age', 'gender', 'occupation', 'zip', ],
    )
    users.to_csv(
        os.path.join(os.path.dirname(__file__), '../resources/movie-lens-100k-users.tsv'), sep='\t', index=False,
    )


if __name__ == '__main__':
    main()
