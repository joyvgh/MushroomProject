''' Joy & Valerie Take The Mushroom Algorithm!! '''
import rule_extraction
import neural_net

def load_data():
	return ''

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
