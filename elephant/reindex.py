import networkx


def main():
    for data_set_name in ['airport', 'authors', 'collaboration', 'facebook', 'congress', 'forum']:
        graph = networkx.read_weighted_edgelist('../graph/' + data_set_name + '.tsv')
        graph = networkx.convert_node_labels_to_integers(graph)
        networkx.write_weighted_edgelist(graph, '../reindexed_graphs/' + data_set_name + '.tsv', delimiter='\t')


if __name__ == '__main__':
    main()
