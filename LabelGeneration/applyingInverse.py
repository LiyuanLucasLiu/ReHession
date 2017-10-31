_author__ = 'S2free'

import argparse
import json


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_json', nargs='+', default=['./Data/intermediate/KBP/ntrain.json'])
	parser.add_argument('--output_json', nargs='+', default=['./Data/intermediate/KBP/ntrain.json'])
	parser.add_argument('--None_name', nargs='+', default=['None'])
	parser.add_argument('--range_min', nargs='+', type=int, default=[0])
	parser.add_argument('--range_max', nargs='+', type=int, default=[148])
	parser.add_argument('--new_none_idx', nargs='+', type=int, default=[156])
	args = parser.parse_args()

	for ind in range(0, len(args.input_json)):
		with open(args.input_json[ind], 'r') as fin:
			lines = fin.readlines()
		with open(args.output_json[ind], 'w') as fout:
			for line in lines:
				ins = json.loads(line)
				rml = ins['relationMentions']
				nrml = list()
				for rm in rml:
					nrm = rm
					label = True
					for anno in rm[3]:
						if anno[1] <= args.range_max[ind] and anno[1] >= args.range_min[ind]:
							label = False
							break
					if label:
						nrm[3].append([args.None_name[ind], args.new_none_idx[ind]])
					nrml.append(nrm)
				ins['relationMentions']=nrml
				fout.write(json.dumps(ins)+'\n')
