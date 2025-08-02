from tree_bh_helpers import treeBH, construct_tree, propagate, get_fisher_p, get_simes_p, returnDF, VisualizeAsSVG, prepare_df

def execute_tree_bh_csv(csv_path, method, alpha, return_type):
    df = prepare_df(csv_path)
    tree = construct_tree(df)
    propagate(tree, get_fisher_p if method == 'fisher' else get_simes_p)
    print(tree)
    treeBH(tree, alpha)

    if return_type == 'df':
        return returnDF(tree)
    elif return_type == 'tree':
        return tree


def execute_tree_bh_tree(tree, method, alpha, return_type):
    propagate(tree, get_fisher_p if method == 'fisher' else get_simes_p)
    print(tree)
    treeBH(tree, alpha)

    if return_type == 'df':
        return returnDF(tree)
    elif return_type == 'tree':
        return tree


def visualize_tree(tree_root, name):
    VisualizeAsSVG(tree_root, name)


def execute_SimesAndFisher_csv(csv_path, alpha,path):
    simes_results = execute_tree_bh_csv(csv_path, 'simes', alpha, 'df')
    fisher_results = execute_tree_bh_csv(csv_path, 'fisher', alpha, 'df')

    combined = simes_results.rename(columns={'rejected': 'rejected_simes', 'pvalue': 'pvalue_simes'})
    combined['rejected_fisher'] = fisher_results['rejected']
    combined['pvalue_fisher'] = fisher_results['pvalue']

    return combined.to_csv(path, index=False)

def execute_SimesAndFisher_tree(tree, alpha,path):
    simes_results = execute_tree_bh_tree(tree, 'simes', alpha, 'df')
    fisher_results = execute_tree_bh_tree(tree, 'fisher', alpha, 'df')

    combined = simes_results.rename(columns={'rejected': 'rejected_simes', 'pvalue': 'pvalue_simes'})
    combined['rejected_fisher'] = fisher_results['rejected']
    combined['pvalue_fisher'] = fisher_results['pvalue']

    return combined.to_csv(path, index=False)


