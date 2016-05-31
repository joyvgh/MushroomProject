''' Joy & Valerie Take on the Mushroom Algorithm!! '''
import multilayernetwork
import numpy as np
import multilayernetwork as mn 

#Global Iris shape values
X1_MED = (5.5, 6.1)
X2_MED = (2.75,3.2)
X3_MED = (2.0, 4.93)
X4_MED = (0.6, 1.7)

def load_data(filename):
    """Loads the Iris data into a matrix
        Shape: sepal length, sepal width, petal length, petal width
        Iris Class: iris-setosa, iris-versicolor, iris-virginica"""
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
    """Given data matrix outputted by load_data, reformat
        data into the proper inputs and expected outputs for
        each line of data. 
    """
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

def make_category_outputs(expected_outputs):
    '''Returns a tuple with the expected outputs for each feature'''
    setosa_outputs = []
    versicolor_outputs = []
    virginica_outputs = []
    outputs = (setosa_outputs, versicolor_outputs, virginica_outputs)
    for output in expected_outputs:
        if output[0] == 1:
            setosa_outputs.append([1])
            versicolor_outputs.append([0])
            virginica_outputs.append([0])
        elif output[1] == 1:
            setosa_outputs.append([0])
            versicolor_outputs.append([1])
            virginica_outputs.append([0])
        else:
            setosa_outputs.append([0])
            versicolor_outputs.append([0])
            virginica_outputs.append([1])
    return outputs


def make_feature_table(weights, feature_labels):
    """Create table that holds Delta values to be used by our algorithm.
        Currently is specific to the Iris dataset. Takes in the weights of
        the network and the labels for each input to the network. Outputs
        a 2D list which holds the Delta values for each label for each
        feature, and a 2D list with the respective labels for those Deltas."""
    table = []
    count = 0
    for i in range(len(feature_labels)):
        row = []
        row.append(weights[count] - weights[count+1] - weights[count+2])
        row.append(-1 * weights[count] + weights[count+1] - weights[count+2])
        row.append(-1 * weights[count] - weights[count+1] + weights[count+2])
        count += 3
        table.append(row)
    
    # condense table, again assuming features are in groups of three originally
    new_feature_labels = []
    for i in range(len(table)):
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
            new_feature_labels.append(["s", "m", "l"])

    return table, new_feature_labels

def traverse_graph(table, feature_labels):
    '''
    Returns a list of rules in the following structure:
    [[(1, 'm'), (2, 's')], [(3, 'l')]] is equivalent to:
    (x_1 = m AND x_2 = s) OR (x_3 = large)
    '''
    rules = []

    # Used to know when the graph can be pruned / whether a rule exists or not:
    max_list = [0 for i in range(len(table))]
    min_list = [0 for i in range(len(table))]

    for i in range(1, len(table)+1):
        max_index = np.argmax(table[-i])
        min_index = np.argmin(table[-i])
        max_list[-i] = table[-i][max_index]
        min_list[-i] = table[-i][min_index]

    for i in range(2, len(table)+1):
        max_list[-i] += max_list[-i+1]
        min_list[-i] += min_list[-i+1]

    graph_helper(table, rules, 0, [], max_list, min_list)

    #prettify rules list
    for i in range(len(rules)):
        for j in range(len(rules[i])):
            row, col = rules[i][j]
            rules[i][j] = (row, feature_labels[row][col])

    return rules



def graph_helper(table, rules, value, cur_path, max_list, min_list):
    """If, given our path, we can create a new rule, add a new rule to
        rules. Otherwise, recurse on all possible paths from current path
    """
    # Get the current row we are looking at
    row = len(cur_path)

    # Base Cases
    # At leaf of tree
    if len(cur_path) >= len(table):
        if (value >= 0):
            return
        else:
            rules.append(cur_path)
            return
    else:
        # Can shortcut to making rule and stop searching tree
        if (value + max_list[row] < 0):
            rules.append(cur_path)
            return
        # Can stop searching tree
        elif (value + min_list[row] > 0):
            return
        
    # Update value and cur_path for each Delta in our current row and
    # continue searching tree
    for i in range(len(table[row])):
        new_value = value + table[row][i]
        cur_copy = cur_path[:]
        cur_copy.append((row, i))
        graph_helper(table, rules, new_value, cur_copy, max_list, min_list)

def percent_correct(output, expected_output):
    ''' Returns the percent of outputs that are correct with the categorization'''
    total = 0.0
    for i in range(len(output)):
        #.8 is arbitrary
        point = 0
        if output[i][0] > 0.8:
            point = 1
        else:
            point = 0
        if(point == expected_output[i][0]):
            total += 1  
    return (total / len(expected_output)) * 100

def rule_to_string(rules):
    '''this is some ugly code to make some pretty string'''
    string = '('
    for i in range(len(rules)):
        for j in range(len(rules[i])):
            string += "(x_" + str(rules[i][j][0]) + "=" + str(rules[i][j][1]) + ") and "
        string = string[:-5]
        string += ") or ("
    string = string[:-4] 
    return string


def main():
    feature_labels = [['s','m','l'],['s','m','l'],['s','m','l'],['s','m','l']]
    categories = ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']
    print("Loading Data...")
    data = load_data('data/iris.data')
    print("Labeling Features...")
    X, general_outputs = label_features(data)
    X = np.array(X).T
    expected_outputs = make_category_outputs(general_outputs)
    print()
    for i in range(len(expected_outputs)):
        print("Starting Category: " + categories[i])
        #Categorization
        Y = np.array(expected_outputs[i]).T
        print("Training Network (this will take a minute...)")
        weights, bias = mn.train_multilayer_network(X, Y, mn.update_weights)
        output = mn.predict_multilayer_network(X, weights, bias, mn.logistic, mn.logistic, 10).T
        print("Percent Correct Categorization: " + str(percent_correct(output, expected_outputs[i])) + "%")

        #Rule Extraction
        print("Rule Extraction!")
        table, labels = make_feature_table(weights[0][0], feature_labels)
        rules = traverse_graph(table, labels)
        print("Rules Found: " + rule_to_string(rules))
        print()

if __name__ == '__main__':
    main()



