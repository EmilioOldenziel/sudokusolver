import sys
import recognize_sudoku
import norvig_sudoku_solver as nss

# print the sudoku
def print_sudoku(sudoku):
	for i in range(0,9):
		for j in range(0,9):
			sys.stdout.write(str(sudoku[i*9+j]))
		sys.stdout.write('\n')


def main():
	# read imagenames from the command line
	image_names = sys.argv[1:]
	for image_name in image_names:
		sudoku = []
		try:
			sudoku = recognize_sudoku.recognize(image_name)
		except:
			print("unable to read sudoku")
			return
		print('The sudoku that was read from the image')
		print_sudoku(sudoku)
		solved_sudoku = nss.solve(str(sudoku))
		print('The solved sudoku:')
		solved_sudoku = [value for (key, value) 
			in sorted(solved_sudoku.items())]
		print_sudoku(solved_sudoku)
main()
