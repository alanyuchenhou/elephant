import os

import pandas


def main():
    users = pandas.read_csv(
        os.path.join(os.path.dirname(__file__), '../resources/u.data'),
        sep='\t',
        names=['user', 'item', 'rating', 'time', ],
    )
    users.to_csv(
        os.path.join(os.path.dirname(__file__), '../resources/movie-lens-100k-ratings.tsv'), sep='\t', index=False,
    )


if __name__ == '__main__':
    main()
