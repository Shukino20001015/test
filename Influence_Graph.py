import networkx as nx
import numpy as np
import pandas as pd
def add_nodes(G, artist_data):
    nodes_list=[]
    for index in artist_data.index:
        row= artist_data.loc[index, :]
        node=(row["artist_id"],{"artist_name":row["artist_name"],"main_genre":row["main_genre"],"active_year":row["active_year"],"popularity":row["popularity"],"count":row["count"],
                  })
        nodes_list.append(node)
    G.add_nodes_from(nodes_list)
    return G
def add_edges(G,influence_data,Grey_index_data):
    # Adjoining edge
    for index in influence_data.index:
        id1=influence_data.loc[index,"follower_id"]
        id2=influence_data.loc[index,"influencer_id"]
        if (id1 in G) and (id2 in G):
            G.add_edge(id1,id2)

    # Obtain the degree neighbor node and calculate the weight of the corresponding edge
    for node_id in G:
        neg_node=G[node_id]
        weight_dict=caculate_influence_weight(G,neg_node,Grey_index_data)
        if weight_dict:
            for neg_name in weight_dict:
                if weight_dict[neg_name]!=0:
                    G.edges[node_id, neg_name]["factor"] = weight_dict[neg_name][0]  # weights
                    G.edges[node_id,neg_name]["weight"]=weight_dict[neg_name][1] # Edge weights
                else:
                    print("Zero appears")
                    G.edges[node_id, neg_name]["factor"] = 0  # weights
                    G.edges[node_id,neg_name]["weight"]=0    # Edge weights

    # Returns the graph
    return G
# Calculate the influence of influencers on followers
def caculate_influence_weight(G,neg_node,Grey_index_data):
    neg_ids=list(neg_node)
    neg_data=Grey_index_data[Grey_index_data["artist_id"].isin(neg_ids)]
    weight_dict = dict()
    r_dict=GCE(neg_data)

    for (index,id) in enumerate(neg_ids):
        pupularity=G.nodes[id].get("popularity")
        if pupularity:
            weight_dict[id]=[r_dict[id],r_dict[id]*pupularity]
        else:
            weight_dict[id]=0
    return weight_dict

count=0
# The weight factor is calculated by using grey correlation
def GCE(neg_data):
    global count
    neg_data.index=neg_data.artist_id
    r_dict=neg_data.loc[:,"artist_id"].to_dict()
    influencer_matrix=np.matrix(neg_data.loc[:, "previous_members_nums":"current_ranking"])

    # Floating point arithmetic
    if influencer_matrix.shape[0]==0:
        return 0
    influencer_matrix = influencer_matrix.astype("float")

    # Normalize the forward matrix
    x_mean = influencer_matrix.mean(axis=0)

    # Minimal index conversion
    influencer_matrix[:,-1]=influencer_matrix[:,-1].max()-influencer_matrix[:,-1]

    for index in range(x_mean.shape[1]):
        if x_mean[0,index]!=0:
            influencer_matrix[:,index]/=x_mean[0,index]
    if np.count_nonzero(np.isnan(influencer_matrix))!=0:
        print(influencer_matrix)
    # Get the reference data column
    x0=influencer_matrix.max(axis=0)

    # Reference data relative matrix
    abs_matirx=abs(influencer_matrix-x0)
    min_min=min(abs_matirx.min(axis=1))
    max_max=max(abs_matirx.max(axis=1))
    cox_matrix=caculate_matrix(abs_matirx,min_min, max_max)
    r_vector=cox_matrix.sum(axis=1)/cox_matrix.shape[0]

    if r_vector.sum()!=0:
        r_vector=r_vector/r_vector.sum()

    for (index, key) in enumerate(r_dict.keys()):
        r_dict[key]=r_vector[index]
    count+=sum(np.isnan(r_vector))
    return r_dict

def caculate_matrix(matrix,min_min,max_max):
    cox_matrix=np.zeros(matrix.shape)

    for (index,array) in enumerate(matrix):
        temp=array + 0.5 * max_max
        temp[np.where(temp==0)]=1
        cox_matrix[index,:]=(min_min+0.5*max_max)/temp
        if np.count_nonzero(np.isnan(cox_matrix)) != 0:
            print(cox_matrix)
    return cox_matrix
# visulize the Graph
def Visua_Graph(G,label):
    import matplotlib.pyplot as plt
    print("come here!")
    nx.write_gexf(G, "data/output/graph/"+label+".gexf")
    # # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw(G)
    # plt.rcParams['savefig.dpi'] =1000 # Image pixels
    # plt.rcParams['figure.dpi'] = 1000 # resolution
    plt.savefig("data/output/"+"Graph.jpg")
    plt.show()

def Main_Graph_Programme():
    G=nx.DiGraph()
    influence_data=pd.read_csv("data/init/"+"influence_data_init.csv")
    artist_data=pd.read_csv("data/init/"+"data_by_artist_init.csv")

    Grey_index_data = pd.read_csv("data/init/" + "Grey_index_data.CSV")
    G=add_nodes(G,artist_data)
    # print(G.nodes(data=True))
    G=add_edges(G,influence_data,Grey_index_data)
    G=G.reverse()       # Inversion, the normal influence relationship is revealed
    print(count)

    weight_output(G)

def weight_output(G):
    output=pd.DataFrame(columns=["artists_id","artists_name","main_genre","influence"])
    for (index,node_id) in enumerate(G):
        influence=G.out_degree(node_id, weight='weight')
        lst=[node_id,G.nodes[node_id]["artist_name"],G.nodes[node_id]["main_genre"],influence]
        output.loc[index,"artists_id":"influence"]=lst
    output.to_csv("data/output/influence/"+"artists_influence.csv")




