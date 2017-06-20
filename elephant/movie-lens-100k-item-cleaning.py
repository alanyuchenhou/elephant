import os

import pandas


def main():
    items = pandas.read_csv(
        os.path.join(os.path.dirname(__file__), '../resources/u.item'),
        sep='|',
        names=[
            'id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
            'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film - Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci - Fi', 'Thriller', 'War',
            'Western',
        ],
        encoding='iso-8859-1',
    )
    items.to_csv(
        os.path.join(os.path.dirname(__file__), '../resources/movie-lens-100k-items.tsv'), sep='\t', index=False,
    )


if __name__ == '__main__':
    main()
