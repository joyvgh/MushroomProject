''' Joy & Valerie Take The Mushroom Algorithm!! '''
import rule_extraction
import neural_net

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
	for line in f:
		X = line.split(',')
		
		if int(X[0]) <= X1_MED[0]:
			X[0] = 's'
		elif int(X[0]) > X1_MED[1]:
			X[0] = 'l'
		else:
			X[0] = 'm'
		

		if int(X[1]) <= X1_MED[0]:
			X[1] = 's'
		elif int(X[1]) > X1_MED[1]:
			X[1] = 'l'
		else:
			X[1] = 'm'
		

		if int(X[2]) <= X1_MED[0]:
			X[2] = 's'
		elif int(X[2]) > X1_MED[1]:
			X[2] = 'l'
		else:
			X[2] = 'm'
		

		if int(X[3]) <= X1_MED[0]:
			X[3] = 's'
		elif int(X[0]) > X1_MED[1]:
			X[3] = 'l'
		else:
			X[3] = 'm'
	return data_matrix

def label_features():
	return None

def main():
	data = load_data()

	label_features()
	neural_net.initialize_network()
	neural_net.train_all_categories()
	graph = rule_extraction.create_graph()
	rules = rule_extraction.get_rules(graph)
	print(rules)
