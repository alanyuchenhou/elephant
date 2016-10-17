import networkx
import numpy


def main():
    graph = networkx.DiGraph(numpy.loadtxt('../resources/House4.mat'))
    with open('../resources/House4.tsv', 'w+') as data_out:
        for edge in networkx.to_edgelist(graph):
            print(edge[0], edge[1], edge[2]['weight'], sep='\t', file=data_out)


if __name__ == '__main__':
    main()
