import pandas


def main():
    links = pandas.read_csv('../resources/' + 'Newman-Cond_mat_95-99-co_occurrence.txt', sep=' ', header=None)
    links.to_csv('../graph/' + 'authors.tsv', sep='\t', index=False, header=False)


if __name__ == '__main__':
    main()
