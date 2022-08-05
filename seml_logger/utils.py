from seml.utils import flatten


def traverse_tree(tree, path='', delimiter='/'):
    if isinstance(tree, dict):
        for k, v in tree.items():
            new_path = path+delimiter+k if len(path) > 0 else k
            for r in traverse_tree(v, new_path, delimiter):
                yield r
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            k = str(i)
            new_path = path+delimiter+k if len(path) > 0 else k
            for r in traverse_tree(v, new_path, delimiter):
                yield r
    elif tree is None:
        return
    else:
        yield path, tree


def construct_suffix(config, naming, delimiter='_'):
    if naming is not None:
        flat_config = flatten(config)
        def to_name(x):
            if x not in flat_config:
                return 'False'
            val = flat_config[x]
            if isinstance(val, (str, bool, int, float)):
                return str(val)
            else:
                return str(val is not None)
        suffix = delimiter + delimiter.join([to_name(n) for n in naming])
    else:
        suffix = ''
    return suffix
