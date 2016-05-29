''' Joy & Valerie Take The Mushroom Algorithm!! '''
import rule_extraction
import neural_net
import multilayernetwork

#Global Iris shape values
X1_MED = (5.5, 6.1)
X2_MED = (2.75,3.2)
X3_MED = (2.0, 4.93)
X4_MED = (0.6, 1.7)

#Loads the Iris data into a matrix
#Shape: sepal length, sepal width, petal length, petal width
#Iris Class: iris-setosa, iris-versicolor, iris-virginica
def load_data(filename):
    f = open(filename, 'r')
    data_matrix = []
    for line in f.read().split('\n'):
        X = line.split(',')
        if float(X[0]) <= X1_MED[0]:
            X[0] = 's'
        elif float(X[0]) > X1_MED[1]:
            X[0] = 'l'
        else:
            X[0] = 'm'
        

        if float(X[1]) <= X2_MED[0]:
            X[1] = 's'
        elif float(X[1]) > X2_MED[1]:
            X[1] = 'l'
        else:
            X[1] = 'm'
        

        if float(X[2]) <= X3_MED[0]:
            X[2] = 's'
        elif float(X[2]) > X3_MED[1]:
            X[2] = 'l'
        else:
            X[2] = 'm'
        

        if float(X[3]) <= X4_MED[0]:
            X[3] = 's'
        elif float(X[3]) > X4_MED[1]:
            X[3] = 'l'
        else:
            X[3] = 'm'

        data_matrix.append(X)
    return data_matrix

def label_features(data_matrix):
    expected_outputs = []
    for line in data_matrix:
        if line[4] == 'Iris-setosa':
            output = [1,0,0]
        elif line[4] == 'Iris-versicolor':
            output = [0,1,0]
        elif line[4] == 'Iris-virginica':
            output = [0,0,1]
        expected_outputs.append(output)
    
    X = []
    for line in data_matrix:
        inputs = []
        if line[0] == 's':
            inputs += [1,-1,-1]
        elif line[0] == 'm':
            inputs += [-1,1,-1]
        else:
            inputs += [-1,-1,1]

        if line[1] == 's':
            inputs += [1,-1,-1]
        elif line[1] == 'm':
            inputs += [-1,1,-1]
        else:
            inputs += [-1,-1,1]

        if line[2] == 's':
            inputs += [1,-1,-1]
        elif line[2] == 'm':
            inputs += [-1,1,-1]
        else:
            inputs += [-1,-1,1]

        if line[3] == 's':
            inputs += [1,-1,-1]
        elif line[3] == 'm':
            inputs += [-1,1,-1]
        else:
            inputs += [-1,-1,1]
        X.append(inputs)
    return X, expected_outputs

def make_feature_table(weights, feature_labels):
    table = []
    count = 0
    for i in range(len(feature_labels)):
        row = []
        row.append(weights[count] - weights[count+1] - weights[count+2])
        row.append(-1 * weights[count] + weights[count+1] - weights[count+2])
        row.append(-1 * weights[count] - weights[count+1] + weights[count+2])
        count += 3 #not general :(
        table.append(row)
    
    # condense table, again assuming features are in groups of three originally
    new_feature_labels = []
    for i in range(len(table)):
        '''
        if table[i][0] == table[i][1]:
            table[i] = [table[i][0], table[i][2]]
            new_feature_labels.append(["not-l", "l"])
        elif table[i][1] == table[i][2]:
            table[i] = [table[i][0], table[i][1]]
            new_feature_labels.append(["s", "not-s"])
        elif table[i][0] == table[i][2]:
            table[i] = [table[i][0], table[i][1]]
            new_feature_labels.append(["not-m", "m"])
        else:
        '''
        new_feature_labels.append(["s", "m", "l"])

    return table, new_feature_labels


def main():
    data = load_data() #done
    label_features()
    
    neural_net.train_all_categories()
    graph = rule_extraction.create_graph()
    rules = rule_extraction.get_rules(graph)
    print(rules)
