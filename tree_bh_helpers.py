import numpy as np
import pandas as pd
import anytree as tree
from statsmodels.stats import multitest as tests
import scipy.stats as stat
from executing import Source
from anytree.exporter import DotExporter
from graphviz import Source


def node_get_children_pvalues(node):
    valid_children = [child for child in node.children if not np.isnan(child.pvalue)]
    child_pvalues = np.array([child.pvalue for child in valid_children])

    return valid_children, child_pvalues

# Bologmov et al. method for tree-based hypothesis testing
def treeBH(tree_root, alpha):
    L = tree_root.height + 1
    qs = np.full(L, alpha)
    for node in tree.PreOrderIter(tree_root):
        node.q_l = qs[node.height]  # Store q^(l) for each node
        node.q_i = 1 

    #############################
    # recursively define q_i by getting rolling rejection rate along node ancestry
    #############################
    def treeBH_recurse(node):
        def recurse(node):
            return not np.isnan(node.id) and node.rejected and not node.is_leaf
    
        # Now we can do a BH correction on the children of the current node
        valid_children, child_pvalues = node_get_children_pvalues(node)
        rejs, _, _, _ = tests.multipletests(child_pvalues, alpha=node.q_i * node.q_l, method='fdr_bh')
        for child, rej in zip(valid_children, rejs):
            child.rejected = rej

        prop_rejected = np.sum(rejs) / len(valid_children) if len(valid_children) > 0 else 0

        for child in node.children:
            child.q_i = node.q_i * prop_rejected
            if recurse(child):
                treeBH_recurse(child)
    
    # Start the recursion from the root node
    treeBH_recurse(tree_root)


def prepare_df(csv_path):
    df =  pd.read_csv(csv_path)

    df = df.dropna(subset=['pvalue']) 

    df['rejected'] = False #create a new column for rejected
    df = df.rename(columns={'Parent ID': 'parent_id', 'Acronym': 'acronym', 'ID' : 'id'})

    #add the root node
    #Fix for real run 
    root_add = {'id': 8, 'parent_id': None, 'acronym': 'grey', 'Name': 'grey', 'parent_acronym': None, 'pvalue': 0, 'rejected': True}
    df = pd.concat([pd.DataFrame([root_add]), df], ignore_index=True)

    return df

def construct_tree(df):
    root_row = df[df['parent_id'].isnull()].iloc[0]
    root = tree.AnyNode(
        id = root_row['id'],
        parent = None,
        acronym = root_row['acronym'],
        name = root_row['Name'],
        parent_acronym = root_row['parent_acronym'],
        pvalue = root_row['pvalue'],
        rejected = root_row['rejected'],
    )

    def add_children(parent_node):
        children_rows = df[df['parent_id'] == parent_node.id]
        for _, row in children_rows.iterrows():
            child_node = tree.AnyNode(
                id = row['id'],
                parent = parent_node,
                acronym = row['acronym'],
                name = row['Name'],
                parent_acronym = row['parent_acronym'],
                pvalue = row['pvalue'],
                rejected = row['rejected'],
            )
            add_children(child_node)

    # Recursively add children nodes to make the tree       
    add_children(root)

    return root 


def propagate(tree_root, method):
    # I am going to do this a dumb way
    for node in tree.PreOrderIter(tree_root):
        if not node.is_leaf:
            node.pvalue = np.nan 
    def descend(node):
        for child in node.children:
            descend(child)
        if node.children:
            node.pvalue = method(node)
    descend(tree_root)

def get_simes_p(node):
    _, child_pvalues = node_get_children_pvalues(node)
    if len(child_pvalues) == 0:
        print(f"Node {node.id} has no valid children p-values and it is {node.is_leaf} that {node.id} is a leaf")
        return np.nan
    else:
        child_pvalues = sorted(child_pvalues)
        m = len(child_pvalues)
        p_value = m * min( child_pvalues[i] / (i + 1) for i in range(m))
        return p_value



def get_fisher_p(node):
    _, child_pvalues = node_get_children_pvalues(node)

    if len(child_pvalues) == 0:
        print(f"Node {node.id} has no valid children p-values and it is {node.is_leaf} that {node.id} is a leaf")
        return np.nan
    else:
        test_stat =  -2 * np.sum(np.log(child_pvalues))
        p_value = stat.chi2.sf(test_stat, 2 * len(child_pvalues))
        return p_value



def VisualizeAsSVG(tree_root, filename):
    DotExporter(
        tree_root,
        nodeattrfunc=lambda node: (
            'shape=circle, style=filled, fillcolor={}, label="{}"'
            .format('pink' if node.rejected else 'white', node.acronym)
        )
    ).to_dotfile("ex_tree.dot")
    Source.from_file("ex_tree.dot").render(f'/Users/evanschwartz/NeuroTree/EvanTreeBH/svgsofTrees/{filename}.svg', format="svg", cleanup=True)


def returnDF(tree_root):
    data = []
    for node in tree.PreOrderIter(tree_root):
        data.append({
            'id': node.id,
            'parent_id': node.parent.id if node.parent else None,
            'acronym': node.acronym,
            'Name': node.name,
            'parent_acronym': node.parent_acronym if node.parent else None,
            'pvalue': node.pvalue,
            'rejected': node.rejected
        })
    df = pd.DataFrame(data)
    return df