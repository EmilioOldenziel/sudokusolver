import sys
import recognize_sudoku
import norvig_sudoku_solver as nss

# print the sudoku
def print_sudoku(sudoku):
    for i in xrange(9):
        for j in xrange(9):
            sys.stdout.write(unicode(sudoku[i*9+j]))
        sys.stdout.write('\n')


def main():
	# read imagenames from the command line
    image_names = sys.argv[1:]
    for image_name in image_names:
    	sudoku = recognize_sudoku.recognize(image_name)
    	print('The sudoku that was read from the image')
    	print_sudoku(sudoku)
    	solved_sudoku = nss.solve(unicode(sudoku))
    	print('The solved sudoku:')
    	solved_sudoku = [value for (key, value) in sorted(solved_sudoku.items())]
    	print_sudoku(solved_sudoku)

main()
