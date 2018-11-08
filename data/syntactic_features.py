
import collections
from allennlp.predictors.predictor import Predictor

PARSER = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")


def parse_dependency_tree(sentence):
    tree = PARSER.predict(sentence=sentence)
    words = tree['words']
    pos = tree['pos']
    deps = tree['predicted_dependencies']
    heads = tree['predicted_heads']
    return {'word': words,
            'pos': pos,
            'dep': deps,
            'head': heads}


def get_children(tree):
    matrix = []
    for i in range(len(tree['head'])):
        children = []
        for j in range(len(tree['head'])):
            if i == j:
                continue
            if tree['head'][j] == i + 1:
                children.append(j)
        matrix.append(tuple(children))
    return matrix


def get_head(tree):
    return tree['head'].index(0)


Node = collections.namedtuple('Node', ['word', 'pos', 'dep'])


def make_tree(tree):
    def make_tree_(data, children, head):
        word, pos, dep = data['word'][head], data['pos'][head], data['dep'][head]
        node = Node(word, pos, dep)
        hchildren = children[head]
        if not hchildren:
            return node
        else:
            return node, tuple(make_tree_(data, children, h) for h in hchildren)

    return make_tree_(tree, get_children(tree), get_head(tree))


def depth_first(root):
    def depth_first_(level, node):
        if isinstance(node, Node):  # leaf
            yield level, node
        else:
            node, children = node
            yield level, node
            for child in children:
                yield from depth_first_(level+1, child)

    return list(depth_first_(1, root))


def print_tree(tree, indent=2):
    for level, node in depth_first(tree):
        print(" " * level * indent + '-' + str(node))


def get_features(sentences):
    features = collections.defaultdict(float)
    pos = []
    for sentence in sentences:
        tree = parse_dependency_tree(sentence)
        pos.extend(tree['pos'])
        parsed = make_tree(tree)
        print()
        print(sentence)
        print_tree(parsed)
        print()
        depths = [depth for depth, _ in depth_first(parsed)]
        features["mean_depth"] += 0 if not depths else sum(depths) / len(depths)
        features["max_depth"] += 0 if not depths else max(depths)
        spans = list(map(len, filter(None, get_children(tree))))
        features["mean_span"] += 0 if not spans else sum(spans) / len(spans)
        features["max_span"] += 0 if not spans else max(spans)
        features["num_spans"] += len(spans)

    # average over sentences
    for feat, val in features.items():
        features[feat] = 0 if not val else val / len(sentences)

    return features
