import os

import pandas
import pycountry


def main():
    countries = pandas.read_csv(
        os.path.join(os.path.dirname(__file__), '../resources/countries.net'),
        sep=' ',
        names=['ID', 'code', ],
        na_filter=False,
    )
    countries['name'] = countries.apply(lambda country: pycountry.countries.get(alpha2=country['code']).name, axis=1, )
    countries = countries.drop('code', axis=1)
    countries.to_csv(
        os.path.join(os.path.dirname(__file__), '../resources/countries.tsv'),
        sep='\t',
        index=False,
    )


if __name__ == '__main__':
    main()
